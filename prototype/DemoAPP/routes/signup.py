from flask import render_template, request, flash, redirect, url_for, session
from werkzeug.security import generate_password_hash
from models import db, User

def signup():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm-password')
        print(f"Received form data: name={name}, email={email}")  # Debug print

        # Basic validation for all fields
        if not name or not email or not password or not confirm_password:
            flash('All fields are required', 'error')
            return render_template('signup.html')

        # Email format validation
        if '@' not in email or '.' not in email:
            flash('Please enter a valid email address', 'error')
            return render_template('signup.html')

        # Name validation (at least 2 characters)
        if len(name.strip()) < 2:
            flash('Please enter a valid full name', 'error')
            return render_template('signup.html')

        # Password validation
        if len(password) < 6:
            flash('Password must be at least 6 characters long', 'error')
            return render_template('signup.html')

        # Password and confirm password must match exactly
        if password != confirm_password:
            flash('Passwords do not match. Please re-enter your confirm password.', 'error')
            return render_template('signup.html')

        # Store user data in information.txt
        import json
        import os
        
        # Get the correct file path relative to the current directory
        file_path = os.path.join(os.path.dirname(__file__), '..', 'information.txt')
        
        try:
            # Try to read existing data
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    data = json.loads(content)
                    full_names = data.get('full_names', [])
                    emails = data.get('emails', [])
                    passwords = data.get('passwords', [])
                else:
                    full_names = []
                    emails = []
                    passwords = []
        except (FileNotFoundError, json.JSONDecodeError):
            full_names = []
            emails = []
            passwords = []
        
        # Check if email already exists
        if email in emails:
            flash('Email already registered', 'error')
            return render_template('signup.html')
        
        # Add new data
        full_names.append(name)
        emails.append(email)
        passwords.append(password)
        
        # Write updated data back to file
        data = {
            'full_names': full_names,
            'emails': emails,
            'passwords': passwords
        }
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Success message
        flash(f'Name and email validated successfully! Welcome {name}!', 'success')
        # Store user info in session for thank you page
        session['registered_name'] = name
        session['registered_email'] = email
        return redirect('/thank-you')
    
    # Handle GET request - show the signup form
    return render_template('signup.html')
