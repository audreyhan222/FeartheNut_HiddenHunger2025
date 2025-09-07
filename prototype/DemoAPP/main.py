from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from datetime import timedelta
from routes.home import home
from routes.form import form
from routes.mission import mission
from routes.articles import articles
from routes.login import login
from routes.signup import signup
from routes.thank_you import thank_you
from routes.logout import logout
from models import db

app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Required for sessions and flash messages
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)  # Session expires after 7 days
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize the database
db.init_app(app)

# Register routes
app.add_url_rule('/', view_func=home)
app.add_url_rule('/form', view_func=form, methods=['GET', 'POST'])
app.add_url_rule('/mission', view_func=mission)
app.add_url_rule('/articles', view_func=articles)
app.add_url_rule('/login', view_func=login, methods=['GET', 'POST'])
app.add_url_rule('/signup', view_func=signup, methods=['GET', 'POST'])
app.add_url_rule('/thank-you', view_func=thank_you)
app.add_url_rule('/logout', view_func=logout)

def create_database():
    with app.app_context():
        print("Creating database...")
        db.create_all()
        print("Database created successfully!")

if __name__ == '__main__':
    create_database()  # Create database tables before running the app
    app.run(debug=True)
