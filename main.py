from flask import Flask, render_template, request, jsonify, redirect, session, url_for
from flask_sqlalchemy import SQLAlchemy
import os

from utils.predict import predict_image
from utils.rag_model import RAGChatbot

app = Flask(__name__)
app.secret_key = "fruitai_secret_key"
# =========================
# CONFIG DATABASE
# =========================
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///fruitai.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# =========================
# CHATBOT
# =========================
bot = RAGChatbot("data.json")


def chatbot_answer(question):
    return bot.generate(question)


# =========================
# UPLOAD FOLDER
# =========================
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# =========================
# DATABASE MODELS
# =========================

# User table
class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)


# Detection history table
class DetectionHistory(db.Model):
    __tablename__ = "detection_history"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer)
    image = db.Column(db.String(255))
    boxed = db.Column(db.String(255))
    fruit_type = db.Column(db.String(100))
    result = db.Column(db.String(100))
    confidence = db.Column(db.Float)
    created_at = db.Column(db.DateTime, server_default=db.func.now())


# Chat history table
class ChatHistory(db.Model):
    __tablename__ = "chat_history"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer)
    question = db.Column(db.Text)
    answer = db.Column(db.Text)
    created_at = db.Column(db.DateTime, server_default=db.func.now())


# Disease table
class Disease(db.Model):
    __tablename__ = "diseases"

    id = db.Column(db.Integer, primary_key=True)
    disease_name = db.Column(db.String(100))
    description = db.Column(db.Text)
    treatment = db.Column(db.Text)


# =========================
# CREATE DATABASE
# =========================
with app.app_context():
    db.create_all()

# =========================
# REGISTER
# =========================
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]

        check_user = User.query.filter_by(email=email).first()

        if check_user:
            return "Email đã tồn tại"

        new_user = User(
            username=username,
            email=email,
            password=password
        )

        db.session.add(new_user)
        db.session.commit()

        return redirect("/login")

    return render_template("register.html")

# =========================
# LOGIN
# =========================
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        user = User.query.filter_by(
            email=email,
            password=password
        ).first()

        if user:
            session["user_id"] = user.id
            session["username"] = user.username
            return redirect("/home")

        return "Sai tài khoản hoặc mật khẩu"

    return render_template("login.html")

# =========================
# LOGOUT
# =========================
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")

# =========================
# HOME
# =========================
@app.route("/")
@app.route("/home")
def home():
    if "user_id" not in session:
        return redirect("/login")

    return render_template("home.html")


# =========================
# DETECT
# =========================
@app.route("/detect", methods=["GET", "POST"])
def detect():
    if "user_id" not in session:
        return redirect("/login")
    result = None
    img_path = None
    heatmap_path = None
    confidence = None

    if request.method == "POST":
        file = request.files["image"]
        fruit_type = request.form.get("fruit_type")

        if file:
            path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(path)

            result, confidence, heatmap_path = predict_image(path, fruit_type)
            img_path = path

            # Save detection history to database
            history = DetectionHistory(
                user_id=session["user_id"],
                image=img_path,
                boxed=heatmap_path,
                fruit_type=fruit_type,
                result=result,
                confidence=float(str(confidence).replace("%", ""))
            )

            db.session.add(history)
            db.session.commit()

    return render_template(
        "index.html",
        result=result,
        img_path=img_path,
        heatmap_path=heatmap_path,
        confidence=confidence
    )


# =========================
# CHATBOT
# =========================
@app.route("/chatbot", methods=["GET", "POST"])
def chatbot():
    if "user_id" not in session:
        return redirect("/login")
    if request.method == "POST":
        question = request.form.get("question")
        answer = chatbot_answer(question)

        if not answer:
            answer = "Xin lỗi, tôi không nhận được phản hồi từ hệ thống AI."

        # Save chat history
        chat = ChatHistory(
            user_id=session["user_id"],
            question=question,
            answer=answer
        )

        db.session.add(chat)
        db.session.commit()

        return jsonify({
            "response": answer
        })

    # Load old chats
    chats = ChatHistory.query.order_by(
        ChatHistory.created_at.desc()
    ).all()

    return render_template("chatbot.html", history=chats)


# =========================
# HISTORY
# =========================
@app.route("/history")
def history():
    if "user_id" not in session:
        return redirect("/login")

    histories = DetectionHistory.query.filter_by(
        user_id=session["user_id"]
    ).order_by(
        DetectionHistory.created_at.desc()
    ).all()

    return render_template("history.html", history=histories)


# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    app.run(debug=True)