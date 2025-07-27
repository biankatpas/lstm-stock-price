from flask import Flask

from routes import health_bp, predict_bp, scrape_bp, status_bp, train_bp

app = Flask(__name__)

# Register blueprints
app.register_blueprint(health_bp)
app.register_blueprint(train_bp)
app.register_blueprint(status_bp)
app.register_blueprint(scrape_bp)
app.register_blueprint(predict_bp)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
