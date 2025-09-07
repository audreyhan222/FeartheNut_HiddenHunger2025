from flask import render_template, session

def articles():
    is_logged_in = 'user_email' in session
    user_email = session.get('user_email', '')
    user_full_name = session.get('user_full_name', '')
    return render_template('articles.html', is_logged_in=is_logged_in, user_email=user_email, user_full_name=user_full_name)
