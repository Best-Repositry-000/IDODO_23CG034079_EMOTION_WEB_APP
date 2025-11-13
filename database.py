from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class UserEmotion(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    image_path = db.Column(db.String(200))
    emotion = db.Column(db.String(50))
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())
