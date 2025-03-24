"""
URL configuration for Agriconnect project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from testapp import views
from testapp.views import logout_view
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('',views.index,name='index'),
    path('get-states/', views.get_states, name='get_states'),
    path('get-districts/', views.get_districts, name='get_districts'),
    path('get-subdistricts/', views.get_subdistricts, name='get_subdistricts'),
    path('get-villages/', views.get_villages, name='get_villages'),
    path('create_user/', views.create_user, name='create_user'),
    path('verify_otp/', views.verify_otp, name='verify_otp'),
    path('login_view/', views.login_view, name='login_view'),
    path('logout/', logout_view, name='logout'),
    path('forgot_password/',views.forgot_password, name='forgot_password'),
    path('send_otp_by_forgot_password/',views.send_otp_by_forgot_password, name='send_otp_by_forgot_password'),
    path('chage_password/', views.change_password, name='change_password'),
     path('shop/',views.shop,name='shop'),
    path('about/',views.about,name='about'),
    path('blog/',views.blog,name='blog'),
    path('contact/',views.contact,name='contact'),
    path('home/',views.index,name='index'),
    path('add-product/', views.add_product, name='add_product'),
    path('delete-product/<int:product_id>/', views.delete_product, name='delete_product'),
    path('edit-product/', views.edit_product, name='edit_product'),
    path('edit-profile/', views.edit_profile, name='edit_profile'),
    # Change password URL
    path('change-password/', views.change_password, name='change_password'),
    path('wishlist/toggle/', views.wishlist_toggle, name='wishlist_toggle'),
    path('wishlist/', views.wishlist, name='wishlist'),
    path('cart', views.cart, name='cart'),
    path('add-to-cart/<int:product_id>/', views.add_to_cart, name='add_to_cart'),
    path('cart/update/', views.update_cart_quantity, name='update_cart_quantity'),
    path('cart/remove/', views.remove_from_cart, name='remove_from_cart'),
    path('quick-view/<int:product_id>/', views.quick_view, name='quick_view'),
    path('update-address/', views.update_address, name='update_address'),
    path('place-order/', views.place_order, name='place_order'),
    path('orders/',views.orders,name="orders"),
    path('submit_review/',views.submit_review,name='submit_review'),
    path('customer_home_page/',views.customer_home_page,name='customer_home_page'),
    path("chat/create/", views.create_or_redirect_chat, name="create_or_redirect_chat"),
    path("chat/", views.chat_room, name="chat_room"),  # Chat home (no specific room)
    path("chat/<str:room_name>/", views.chat_room, name="chat_room"),
    path('mark-seen/', views.mark_seen, name='mark_seen'),
    path('get-chat-history/', views.get_chat_history, name='get_chat_history'),



    
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)