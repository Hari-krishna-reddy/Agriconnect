from django.contrib import admin
from .models import State, District, SubDistrict, Village,Profile

@admin.register(State)
class StateAdmin(admin.ModelAdmin):
    list_display = ('id', 'name')  # Display all fields
    search_fields = ('name',)  # Enable search by name
    ordering = ('name',)  # Order by name

@admin.register(District)
class DistrictAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'state')  # Display all fields
    search_fields = ('name', 'state__name')  # Enable search by district and state name
    list_filter = ('state',)  # Filter by state
    ordering = ('name',)  # Order by name

@admin.register(SubDistrict)
class SubDistrictAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'district')  # Display all fields
    search_fields = ('name', 'district__name')  # Enable search by subdistrict and district name
    list_filter = ('district',)  # Filter by district
    ordering = ('name',)  # Order by name

@admin.register(Village)
class VillageAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'subdistrict')  # Display all fields
    search_fields = ('name', 'subdistrict__name')  # Enable search by village and subdistrict name
    list_filter = ('subdistrict',)  # Filter by subdistrict
    ordering = ('name',)  # Order by name

from django.contrib import admin
from .models import Profile

class ProfileAdmin(admin.ModelAdmin):
    # Specify which fields should be displayed in the list view
    list_display = ('id','user', 'phone_number', 'state', 'district', 'subdistrict', 'village','role')  
   
    # Allow searching by user name, phone number, and location-related fields
    search_fields = ('user__username', 'user__email', 'phone_number', 'state', 'district', 'subdistrict', 'village')  
    
    # Filter options to filter by state, district, and subdistrict
    list_filter = ('state', 'district', 'subdistrict', 'village')  
    
    # Order the profiles by user name
    ordering = ('user__username',)  # Order by user

    # Enable editing of these fields directly in the list view
    readonly_fields = ('user',)  # Make user field readonly, as it is a foreign key

# Register the Profile model with the custom ProfileAdmin
admin.site.register(Profile, ProfileAdmin)



from django.contrib import admin
from .models import Product, ProductImage, Order, OrderItem, Wishlist, AddToCart, ShippingAddress, PaymentStatus, ProductReview

# ProductAdmin to display all fields in the admin panel
class ProductAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'seller', 'price_per_unit', 'quantity_in_stock', 'category', 'unit', 'created_at')
    search_fields = ('name', 'seller__username', 'category')
    list_filter = ('category', 'unit', 'created_at')
    fields = ('seller', 'name', 'description', 'price_per_unit', 'quantity_in_stock', 'category', 'unit', 'created_at')

class ProductImageAdmin(admin.ModelAdmin):
    list_display = ('id', 'product_id', 'product_name', 'image', 'created_at')
    search_fields = ('product__id', 'product__name')  # Allow search by both product ID and product name

    # Method to display product ID
    def product_id(self, obj):
        return obj.product.id
    product_id.admin_order_field = 'product__id'  # Allows sorting by product ID
    product_id.short_description = 'Product ID'  # Column header for product ID

    # Method to display product name
    def product_name(self, obj):
        return obj.product.name
    product_name.admin_order_field = 'product__name'  # Allows sorting by product name
    product_name.short_description = 'Product Name'  # Column header for product name



@admin.register(Order)
class OrderAdmin(admin.ModelAdmin):
    list_display = ('id', 'customer', 'status', 'total_price', 'placed_at')
    list_filter = ('status', 'placed_at')
    search_fields = ('customer__username', 'shipping_details')
    ordering = ('-placed_at',)

class OrderItemAdmin(admin.ModelAdmin):
    list_display = ('id', 'order', 'product', 'quantity', 'price_at_purchase')
    search_fields = ('order__id', 'product__name')

class WishlistAdmin(admin.ModelAdmin):
    list_display = ('id', 'customer', 'product','product_id', 'added_at')
    search_fields = ('customer__username', 'product__name')
     # Optionally, you can define a custom method for displaying product_id if needed:
    def product_id(self, obj):
        return obj.product.id
    product_id.admin_order_field = 'product'  # Allow ordering by product
    product_id.short_description = 'Product ID'  # Display label for the column

class AddToCartAdmin(admin.ModelAdmin):
    list_display = ('id', 'customer', 'product', 'quantity', 'added_at')
    search_fields = ('customer__username', 'product__name')

# Register ShippingAddress model
class ShippingAddressAdmin(admin.ModelAdmin):
    list_display = ('user', 'phone_number', 'address_line1', 'address_line2', 'state', 'district', 'subdistrict', 'village', 'postal_code')
    search_fields = ('user__username', 'phone_number', 'state', 'district', 'village')
    list_filter = ('state', 'district')


class PaymentStatusAdmin(admin.ModelAdmin):
    list_display = ('id', 'order', 'status', 'transaction_id', 'payment_date')
    search_fields = ('order__id', 'transaction_id')

class ProductReviewAdmin(admin.ModelAdmin):
    list_display = ('id', 'product', 'customer', 'rating', 'created_at')
    search_fields = ('product__name', 'customer__username')
    list_filter = ('rating', 'created_at')

# Register models to admin site
admin.site.register(Product, ProductAdmin)
admin.site.register(ProductImage, ProductImageAdmin)
admin.site.register(OrderItem, OrderItemAdmin)
admin.site.register(Wishlist, WishlistAdmin)
admin.site.register(AddToCart, AddToCartAdmin)
admin.site.register(ShippingAddress, ShippingAddressAdmin)
admin.site.register(PaymentStatus, PaymentStatusAdmin)
admin.site.register(ProductReview, ProductReviewAdmin)



from django.contrib import admin
from .models import ChatRoom, Message

@admin.register(ChatRoom)
class ChatRoomAdmin(admin.ModelAdmin):
    list_display = ("name",)  
    search_fields = ("name",)
    filter_horizontal = ("users",)  # Allows selecting multiple users

@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ("sender", "room", "content", "timestamp", "is_seen")
    list_filter = ("room", "timestamp", "is_seen")
    search_fields = ("sender__username", "content")





