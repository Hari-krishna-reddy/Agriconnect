import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack  # Import this
from testapp.routing import websocket_urlpatterns  # Import your WebSocket routes

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "Agriconnect.settings")

application = ProtocolTypeRouter(
    {
        "http": get_asgi_application(),
        "websocket": AuthMiddlewareStack(  # Wrap WebSocket with authentication
            URLRouter(websocket_urlpatterns)
        ),
    }
)
