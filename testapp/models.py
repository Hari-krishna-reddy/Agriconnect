from django.db import models

class State(models.Model):
    name = models.CharField(max_length=100)
    
    def __str__(self):
        return self.name

class District(models.Model):
    name = models.CharField(max_length=100)
    state = models.ForeignKey(State, related_name='districts', on_delete=models.CASCADE)
    
    def __str__(self):
        return self.name

class SubDistrict(models.Model):
    name = models.CharField(max_length=100)
    district = models.ForeignKey(District, related_name='subdistricts', on_delete=models.CASCADE)
    
    def __str__(self):
        return self.name

class Village(models.Model):
    name = models.CharField(max_length=100)
    subdistrict = models.ForeignKey(SubDistrict, related_name='villages', on_delete=models.CASCADE)
    
    def __str__(self):
        return self.name


# models.py
from django.db import models
from django.contrib.auth.models import User
class Profile(models.Model):
    ROLE_CHOICES = [
        ('customer', 'Customer'),
        ('seller', 'Seller'),
    ]

    user = models.OneToOneField(User, on_delete=models.CASCADE)
    state = models.CharField(max_length=100)
    district = models.CharField(max_length=100)
    subdistrict = models.CharField(max_length=100)
    village = models.CharField(max_length=100)
    phone_number = models.CharField(max_length=15)
    role = models.CharField(max_length=10, choices=ROLE_CHOICES, default='customer')

    def __str__(self):
        return self.user.username


class Product(models.Model):
    UNIT_CHOICES = [
        ('kg', 'Kilogram'),
        ('liter', 'Liter'),
        ('dozen', 'Dozen'),
        ('unit', 'Unit'),
    ]

    CATEGORY_CHOICES = [
        ('fruits', 'Fruits'),
        ('vegetables', 'Vegetables'),
        ('grains', 'Grains'),
        ('dairy', 'Dairy'),
        ('meat', 'Meat'),
        ('beverages', 'Beverages'),
        ('snacks', 'Snacks'),
        ('spices', 'Spices'),
        ('others', 'Others'),
    ]
    
    seller = models.ForeignKey(User, on_delete=models.CASCADE)  # Seller is a user
    name = models.CharField(max_length=255)
    description = models.TextField()
    price_per_unit = models.DecimalField(max_digits=10, decimal_places=2)
    quantity_in_stock = models.PositiveIntegerField()
    category = models.CharField(max_length=100, choices=CATEGORY_CHOICES)
    unit = models.CharField(max_length=10, choices=UNIT_CHOICES)  # Field for unit type
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name
    

class ProductImage(models.Model):
    product = models.ForeignKey(Product, related_name='images', on_delete=models.CASCADE)
    image = models.ImageField(upload_to='products/')  # Stored under media/products/
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Image for {self.product.id}"
    

from django.db import models
from django.contrib.auth.models import User

class Order(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('completed', 'Completed'),
        ('canceled', 'Canceled')
    ]

    customer = models.ForeignKey(
        User, related_name='orders', on_delete=models.CASCADE, db_index=True
    )

    # Store phone number and address as a single formatted string
    shipping_details = models.TextField()

    total_price = models.DecimalField(max_digits=10, decimal_places=2)
    status = models.CharField(
        max_length=50, choices=STATUS_CHOICES, default='pending', db_index=True
    )
    placed_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Order {self.id} by {self.customer.username}"



class OrderItem(models.Model):
    order = models.ForeignKey(Order, related_name='items', on_delete=models.CASCADE)
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    quantity = models.PositiveIntegerField()
    price_at_purchase = models.DecimalField(max_digits=10, decimal_places=2)

    def __str__(self):
        return f"{self.product.name} x {self.quantity} in Order {self.order.id}"

class Wishlist(models.Model):
    customer = models.ForeignKey(User, related_name='wishlist', on_delete=models.CASCADE)
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    added_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.product.name} in {self.customer.username}'s wishlist"

class AddToCart(models.Model):
    customer = models.ForeignKey(User, related_name='cart', on_delete=models.CASCADE)
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    quantity = models.PositiveIntegerField(default=1)
    added_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.quantity} of {self.product.name} in {self.customer.username}'s cart"

class ShippingAddress(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    phone_number = models.CharField(max_length=15)
    address_line1 = models.CharField(max_length=255, blank=True, null=True)  # Optional
    address_line2 = models.CharField(max_length=255, blank=True, null=True)  # Optional
    state = models.CharField(max_length=100)  # CharField instead of ForeignKey
    district = models.CharField(max_length=100)  # CharField instead of ForeignKey
    subdistrict = models.CharField(max_length=100)  # CharField instead of ForeignKey
    village = models.CharField(max_length=100)  # CharField instead of ForeignKey
    postal_code = models.CharField(max_length=10, blank=True, null=True)  # Optional

    def __str__(self):
        return f"{self.address_line1 or ''}, {self.address_line2 or ''}, {self.village}, {self.subdistrict}, {self.district}, {self.state}"



class PaymentStatus(models.Model):
    order = models.OneToOneField(Order, related_name='payment_status', on_delete=models.CASCADE)
    status = models.CharField(max_length=50, choices=[('pending', 'Pending'), ('completed', 'Completed'), ('failed', 'Failed')], default='pending')
    transaction_id = models.CharField(max_length=255, blank=True, null=True)  # Payment gateway transaction ID
    payment_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Payment for Order {self.order.id} - {self.status}"


class ProductReview(models.Model):
    product = models.ForeignKey(Product, related_name='reviews', on_delete=models.CASCADE)
    customer = models.ForeignKey(User, on_delete=models.CASCADE)
    rating = models.PositiveIntegerField(choices=[(i, i) for i in range(1, 6)])  # 1-5 rating scale
    comment = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Review by {self.customer.username} on {self.product.name}"
    



from django.db import models
from django.contrib.auth.models import User

class ChatRoom(models.Model):
    name = models.CharField(max_length=255, unique=True)
    users = models.ManyToManyField(User)

class Message(models.Model):
    sender = models.ForeignKey(User, on_delete=models.CASCADE)
    room = models.ForeignKey(ChatRoom, on_delete=models.CASCADE)
    content = models.TextField(blank=True, null=True)
    file = models.FileField(upload_to="chat_files/", blank=True, null=True)
    is_seen = models.BooleanField(default=False)
    timestamp = models.DateTimeField(auto_now_add=True)

    def mark_as_seen(self):
        self.is_seen = True
        self.save()


