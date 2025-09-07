from flask import render_template, session

def thank_you():
    # Get user info from session
    name = session.get('registered_name', 'User')
    email = session.get('registered_email', '')
    
    return render_template('thank_you.html', name=name, email=email)
