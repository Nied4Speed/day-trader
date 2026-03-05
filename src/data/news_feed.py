"""Alpaca news feed service.

Fetches news articles from Alpaca's News API, scores them for sentiment,
persists to SQLite, and provides per-symbol rolling sentiment scores.
"""

import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional

from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import NewsRequest

from src.core.config import Config
from src.core.database import NewsArticle, get_session
from src.data.sentiment import score_article

logger = logging.getLogger(__name__)


class AlpacaNewsFeed:
    """Fetches and scores Alpaca news for sentiment signals."""

    def __init__(self, config: Config):
        self.config = config
        self._client: Optional[NewsClient] = None
        # symbol -> list of (timestamp, score)
        self._recent_scores: dict[str, list[tuple[datetime, float]]] = defaultdict(list)

    @property
    def client(self) -> NewsClient:
        if self._client is None:
            self._client = NewsClient(
                api_key=self.config.alpaca.api_key,
                secret_key=self.config.alpaca.secret_key,
            )
        return self._client

    def fetch_news(
        self,
        symbols: list[str],
        lookback_minutes: int = 30,
    ) -> dict[str, float]:
        """Fetch recent news and return per-symbol sentiment scores.

        Returns a dict mapping symbol -> sentiment score in [-1.0, +1.0].
        Symbols with no recent news are omitted from the result.
        """
        now = datetime.now(timezone.utc)
        start = now - timedelta(minutes=lookback_minutes)

        try:
            request = NewsRequest(
                symbols=",".join(symbols) if isinstance(symbols, list) else symbols,
                start=start,
                end=now,
                limit=50,
                sort="desc",
            )
            response = self.client.get_news(request)
        except Exception:
            logger.exception("Failed to fetch news from Alpaca")
            return {}

        articles = response.data.get("news", []) if hasattr(response, "data") and isinstance(response.data, dict) else []
        if not articles:
            return {}

        db = get_session(self.config.db_path)
        new_count = 0
        try:
            for article in articles:
                # Support both dict and object access
                _get = (lambda a, k, d=None: a.get(k, d)) if isinstance(article, dict) else (lambda a, k, d=None: getattr(a, k, d))

                article_id = str(_get(article, "id", ""))
                if not article_id:
                    continue

                # Deduplicate by alpaca_news_id
                existing = (
                    db.query(NewsArticle)
                    .filter(NewsArticle.alpaca_news_id == article_id)
                    .first()
                )
                if existing:
                    continue

                headline = _get(article, "headline", "") or ""
                summary = _get(article, "summary", "") or ""
                sentiment = score_article(headline, summary)
                article_symbols = _get(article, "symbols", []) or []

                db_article = NewsArticle(
                    alpaca_news_id=article_id,
                    headline=headline,
                    summary=summary,
                    source=_get(article, "source", "") or "",
                    symbols=article_symbols,
                    sentiment_score=sentiment,
                    published_at=_get(article, "created_at", now),
                    fetched_at=now,
                )
                db.add(db_article)
                new_count += 1

                # Update in-memory rolling scores
                for sym in article_symbols:
                    if sym in symbols:
                        self._recent_scores[sym].append(
                            (article.created_at, sentiment)
                        )

            db.commit()
            if new_count:
                logger.info(f"Persisted {new_count} new news articles")
        except Exception:
            db.rollback()
            logger.exception("Failed to persist news articles")
        finally:
            db.close()

        # Prune old scores and compute averages
        cutoff = now - timedelta(minutes=lookback_minutes)
        result: dict[str, float] = {}

        for symbol in symbols:
            scores = self._recent_scores.get(symbol, [])
            # Keep only recent
            scores = [(ts, s) for ts, s in scores if ts >= cutoff]
            self._recent_scores[symbol] = scores

            if scores:
                avg = sum(s for _, s in scores) / len(scores)
                result[symbol] = avg

        return result

    def get_sentiment(self, symbol: str) -> Optional[float]:
        """Get the current rolling sentiment for a symbol, or None."""
        scores = self._recent_scores.get(symbol, [])
        if not scores:
            return None
        return sum(s for _, s in scores) / len(scores)
