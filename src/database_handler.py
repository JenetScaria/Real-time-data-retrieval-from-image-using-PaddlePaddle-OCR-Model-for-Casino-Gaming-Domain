import psycopg2
from psycopg2.extras import RealDictCursor
import logging
from typing import Dict

class DatabaseHandler:
    def __init__(self, db_config: Dict):
        self.db_config = db_config
        self.connection = None
        self.connect()
    
    def connect(self):
        try:
            self.connection = psycopg2.connect(
                host=self.db_config['host'],
                port=self.db_config['port'],
                database=self.db_config['name'],
                user=self.db_config['user'],
                password=self.db_config['password']
            )
            logging.info("Database connection established")
        except Exception as e:
            logging.error(f"Database connection failed: {e}")
            raise
    
    def update_prize(self, machine_id: str, prize_amount: float, confidence: float, timestamp: float):
        query = """
        INSERT INTO slot_machine_prizes (machine_id, prize_amount, confidence, detected_at)
        VALUES (%s, %s, %s, TO_TIMESTAMP(%s))
        ON CONFLICT (machine_id) 
        DO UPDATE SET 
            prize_amount = EXCLUDED.prize_amount,
            confidence = EXCLUDED.confidence,
            detected_at = EXCLUDED.detected_at,
            updated_at = CURRENT_TIMESTAMP;
        """
        
        with self.connection.cursor() as cursor:
            cursor.execute(query, (machine_id, prize_amount, confidence, timestamp))
            self.connection.commit()