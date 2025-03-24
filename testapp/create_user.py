import os
import django
import random
from django.contrib.auth.models import User

# Set up Django environment
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "Agriconnect.settings")  # Replace 'your_project' with your project name
django.setup()

# Sample Indian first and last names
first_names = ["Aarav", "Vivaan", "Aditya", "Rohan", "Krishna", "Priya", "Ananya", "Meera", "Sneha", "Siddharth"]
last_names = ["Sharma", "Verma", "Reddy", "Yadav", "Pillai", "Das", "Roy", "Bose", "Mishra", "Naidu"]

# Generate 500 users
for i in range(1, 501):
    first_name = random.choice(first_names)
    last_name = random.choice(last_names)
    username = f"{first_name.lower()}{last_name.lower()}{i}"  # Unique username
    email = f"{username}@example.com"
    password = "kohli@123"

    if not User.objects.filter(username=username).exists():
        User.objects.create_user(
            username=username,
            email=email,
            first_name=first_name,
            last_name=last_name,
            password=password
        )
        print(f"User {username} created.")
    else:
        print(f"User {username} already exists. Skipping...")
