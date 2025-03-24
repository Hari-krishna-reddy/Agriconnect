import os
import django

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Agriconnect.settings')
django.setup()
from django.contrib.auth.models import User

user = User.objects.get(username="kittu")
print(user.date_joined)  # Outputs the date and time when the user was created
