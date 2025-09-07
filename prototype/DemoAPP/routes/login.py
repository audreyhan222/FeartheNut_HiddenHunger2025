from flask import render_template, request, flash, redirect, url_for, session
import json
import os

def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Get the correct file path relative to the current directory
        file_path = os.path.join(os.path.dirname(__file__), '..', 'information.txt')
        
        try:
            # Try to read existing data
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    data = json.loads(content)
                    emails = data.get('emails', [])
                    passwords = data.get('passwords', [])
                    full_names = data.get('full_names', [])
                else:
                    emails = []
                    passwords = []
                    full_names = []
        except (FileNotFoundError, json.JSONDecodeError):
            emails = []
            passwords = []
            full_names = []
        
        # Check if email and password match at the same index
        login_successful = False
        user_full_name = ""
        
        for i in range(len(emails)):
            if emails[i] == email and passwords[i] == password:
                login_successful = True
                user_full_name = full_names[i] if i < len(full_names) else email
                break
        
        if login_successful:
            # Login successful - redirect to home with success message
            session['user_email'] = email
            session['user_full_name'] = user_full_name
            flash('You Successfully Login', 'success')
            return redirect(url_for('home'))
        else:
            # Login failed - stay on login page with error message
            flash('Username or password does not match, please enter your information again.', 'error')
            return render_template('login.html')
    
    return render_template('login.html')
