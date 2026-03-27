"""
database.py
============
MongoDB integration for storing customer predictions.

- Connects to MongoDB (default: localhost:27017)
- Database: customer_db
- Collection: predictions
- Stores: input data, predicted cluster, timestamp

If MongoDB is not available, operations fail gracefully with logging.
"""

import os
import logging
from datetime import datetime
from typing import Optional

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

# Configure logging
logger = logging.getLogger(__name__)

# MongoDB connection settings
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = "customer_db"
COLLECTION_NAME = "predictions"


class Database:
    """
    Singleton-style MongoDB connection manager.
    Handles connection, insertion, and graceful error handling.
    """

    def __init__(self):
        self.client: Optional[MongoClient] = None
        self.db = None
        self.collection = None
        self._connected = False

    def connect(self):
        """
        Establish connection to MongoDB.
        If the connection fails, the app still runs (predictions won't be stored).
        """
        try:
            self.client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
            # Test the connection
            self.client.admin.command("ping")
            self.db = self.client[DB_NAME]
            self.collection = self.db[COLLECTION_NAME]
            self._connected = True
            logger.info(f"[DB] Connected to MongoDB at {MONGO_URI}")
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            self._connected = False
            logger.warning(
                f"[DB] Could not connect to MongoDB at {MONGO_URI}. "
                f"Predictions will NOT be stored. Error: {e}"
            )

    @property
    def is_connected(self) -> bool:
        """Check if the database is currently connected."""
        return self._connected

    def store_prediction(self, input_data: dict, cluster: int, category: str) -> bool:
        """
        Store a prediction record in the MongoDB collection.

        Args:
            input_data: The customer input data (dict)
            cluster: The predicted cluster number
            category: The human-readable category label

        Returns:
            True if stored successfully, False otherwise
        """
        if not self._connected:
            logger.warning("[DB] Not connected — skipping prediction storage.")
            return False

        try:
            record = {
                "input_data": input_data,
                "predicted_cluster": cluster,
                "category": category,
                "timestamp": datetime.utcnow(),
            }
            self.collection.insert_one(record)
            logger.info(f"[DB] Prediction stored: cluster={cluster}, category={category}")
            return True
        except Exception as e:
            logger.error(f"[DB] Failed to store prediction: {e}")
            return False

    def close(self):
        """Close the MongoDB connection."""
        if self.client:
            self.client.close()
            self._connected = False
            logger.info("[DB] MongoDB connection closed.")


# Global database instance
db = Database()
