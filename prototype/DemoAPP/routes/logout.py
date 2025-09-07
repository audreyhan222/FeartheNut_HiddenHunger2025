from flask import redirect, url_for, session, flash

def logout():
    # Clear the user session
    session.pop('user_email', None)
    flash('You have been successfully logged out', 'success')
    return redirect(url_for('home'))
