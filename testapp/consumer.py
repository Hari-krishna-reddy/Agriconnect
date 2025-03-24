import json
from channels.generic.websocket import AsyncWebsocketConsumer
from django.contrib.auth.models import User
from asgiref.sync import sync_to_async
from .models import ChatRoom, Message
from datetime import datetime


class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        """Handles WebSocket connection."""
        self.room_name = self.scope['url_route']['kwargs']['room_name']
        self.room_group_name = f'chat_{self.room_name}'
        
        print(f"WebSocket Connected to Room: {self.room_name}")  # Debug print
        
        # Join room group
        await self.channel_layer.group_add(self.room_group_name, self.channel_name)
        await self.accept()

        # Send alert to frontend
        await self.send(text_data=json.dumps({"type": "alert", "message": "WebSocket connected!"}))

    async def disconnect(self, close_code):
        """Handles WebSocket disconnection."""
        print(f"WebSocket Disconnected from Room: {self.room_name}")  # Debug print
        # Leave room group
        await self.channel_layer.group_discard(self.room_group_name, self.channel_name)

    async def receive(self, text_data):
        """Handles receiving messages over WebSocket."""
        data = json.loads(text_data)
        print(f"Received data: {data}")  # Debugging print

        sender = self.scope["user"]  # Ensure user is authenticated
        message_type = data.get("type")

        # Ignore "typing" events
        if message_type == "typing":
            print(f"{sender.username} is typing...")  # Debugging print
            return  # Do not process further

        # Mark messages as seen when user opens the chat
        if message_type == "seen":
            await self.mark_messages_as_seen(sender)
            return

        # Process only actual chat messages
        if message_type == "chat_message":
            try:
                room = await sync_to_async(ChatRoom.objects.get)(name=self.room_name)
                content = data.get("message", "").strip()  # Ensure message is fetched properly
                
                print(f"Message Content: '{content}' from {sender.username}")  # Debugging print

                if not content:
                    print("Warning: Empty message received, not saving.")  # Debugging print
                    return

                # Save the message to the database
                message = await self.save_message(sender, room, content)
                if not message:
                    raise ValueError("Failed to save message.")

                # Broadcast message to group
                await self.channel_layer.group_send(
                    self.room_group_name,
                    {
                        "type": "chat_message",
                        "username": sender.username,
                        "message": message.content,
                        "timestamp": message.timestamp.isoformat(),
                        "is_seen": message.is_seen  # Use the correct field name
                    }
                )
            except Exception as e:
                print(f"Error: {e}")  # Debugging print

    async def chat_message(self, event):
        """Broadcasts a chat message to all users in the room."""
        print(f"Broadcasting Message: {event['message']} from {event['username']}")  # Debug print
        # Send message to WebSocket
        await self.send(text_data=json.dumps(event))

    async def save_message(self, sender, room, content):
        """Save the chat message to the database."""
        try:
            message = await sync_to_async(Message.objects.create)(
                sender=sender,
                room=room,
                content=content,
                is_seen=False  # Default: message is not seen initially
            )
            print(f"Message Saved: {message.content} by {message.sender.username}")  # Debug print
            return message
        except Exception as e:
            print(f"Error Saving Message: {e}")  # Debug print
            return None

    async def mark_messages_as_seen(self, user):
        """Marks all unread messages in the room as seen when the user opens the chat."""
        try:
            # Filter messages in the current room that are not sent by the user and are unread
            unseen_messages = await sync_to_async(lambda: Message.objects.filter(
                room__name=self.room_name,
                is_seen=False
            ).exclude(sender=user))()

            # Mark messages as seen
            for msg in unseen_messages:
                msg.is_seen = True
                await sync_to_async(msg.save)()
            print(f"All messages marked as seen for {user.username}")  # Debug print
        except Exception as e:
            print(f"Error updating seen messages: {e}")  # Debug print