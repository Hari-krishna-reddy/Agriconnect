from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.core.mail import send_mail
from django.conf import settings
from .models import Profile,State,District,SubDistrict,Village,Wishlist
import random
from django.contrib.auth.hashers import make_password
from django.http import JsonResponse
from django.core.validators import validate_email
from django.core.exceptions import ValidationError
from django.contrib.auth import authenticate, login
from django.contrib import messages
from django.core.exceptions import ValidationError
from django.core.validators import EmailValidator
from django.contrib.auth import logout
from django.contrib import messages
from django.contrib.auth import update_session_auth_hash
from django.shortcuts import get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from .models import AddToCart, Product,ShippingAddress
from django.views.decorators.csrf import csrf_exempt





import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Load Data from CSV Files
# Replace 'path_to_product_data.csv' and 'path_to_user_interaction_data.csv' with your actual file paths
product_data = pd.read_csv('product_data.csv')
user_interaction_data = pd.read_csv('user_interaction_data.csv')

# Step 2: Feature Engineering
# Interaction Score
user_interaction_data['interaction_score'] = (
    user_interaction_data['views_user'] * 1 +
    user_interaction_data['add_to_cart'] * 3 +
    user_interaction_data['wishlist_adds_user'] * 2 +
    user_interaction_data['purchase'] * 5
)

# Conversion Rates
user_interaction_data['purchase_rate'] = user_interaction_data['purchase'] / (user_interaction_data['views_user'] + 1)
user_interaction_data['cart_conversion_rate'] = user_interaction_data['add_to_cart'] / (user_interaction_data['views_user'] + 1)
user_interaction_data['wishlist_conversion_rate'] = user_interaction_data['wishlist_adds_user'] / (user_interaction_data['views_user'] + 1)

# Fill NaN values
user_interaction_data.fillna(0, inplace=True)

# Merging DataFrames
merged_data = pd.merge(user_interaction_data, product_data, on='product_id', suffixes=('_user', '_product'))

# Normalizing Features
scaler = StandardScaler()
features_to_scale = ['sales', 'rating', 'review_count', 'views', 'added_to_cart', 'wishlist_adds_user',
                     'purchase_rate', 'cart_conversion_rate', 'wishlist_conversion_rate']
merged_data[features_to_scale] = scaler.fit_transform(merged_data[features_to_scale])

# Step 3: Preparing Data for Classification
median_score = merged_data['interaction_score'].median()
merged_data['interaction_label'] = (merged_data['interaction_score'] > median_score).astype(int)  # 1 = High, 0 = Low

X = merged_data[features_to_scale]
y = merged_data['interaction_label']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Model Training (XGBoost Only)
xgb_classifier = xgb.XGBClassifier(n_estimators=100, max_depth=5, scale_pos_weight=1.5, random_state=42)
xgb_classifier.fit(X_train, y_train)
y_pred_xgb = xgb_classifier.predict(X_test)

# Model Evaluation Function
def evaluate_model(y_test, y_pred, model_name):
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"{model_name} Performance:")
    print(f"🔹 Accuracy: {acc:.4f}")
    print(f"🔹 Precision: {prec:.4f}")
    print(f"🔹 Recall: {rec:.4f}")
    print(f"🔹 F1-score: {f1:.4f}")

    return acc, prec, rec, f1

evaluate_model(y_test, y_pred_xgb, "XGBoost")

# Step 5: Content-Based Filtering (Cosine Similarity)
product_features = product_data[['category', 'sales', 'rating', 'review_count', 'views']]
product_features = pd.get_dummies(product_features, drop_first=True)
cosine_sim_content = cosine_similarity(product_features)

# Recommendation System
def popularity_based_recommendation(user_address=None, top_n=20):
    if user_address:
        # Filter products available in the user's region
        filtered_products = product_data[product_data['product_availability'] == user_address]
        return filtered_products.nlargest(top_n, 'wishlist_adds_product')[['product_id', 'rating']]
    else:
        # Default to global popularity if no address is provided
        return product_data.nlargest(top_n, 'wishlist_adds_product')[['product_id', 'rating']]

def content_based_recommendation(new_product_id, user_address=None, top_n=20):
    if new_product_id not in product_data['product_id'].values:
        return "Product not found"

    product_index = product_data[product_data['product_id'] == new_product_id].index[0]
    similar_products = np.argsort(cosine_sim_content[product_index])[::-1]

    if user_address:
        # Filter similar products available in the user's region
        similar_products = [idx for idx in similar_products if product_data.iloc[idx]['product_availability'] == user_address]

    return product_data.iloc[similar_products[:top_n]]

def collaborative_filtering_recommendation(user_id, user_address=None, top_n=20):
    if user_id not in user_interaction_data['user_id'].values:
        return "User not found"

    # Get user's historical data
    user_history = merged_data[merged_data['user_id'] == user_id]
    if user_history.empty:
        return "No historical data found for this user."

    # Get the exact features used during training
    trained_features = xgb_classifier.get_booster().feature_names
    user_features = user_history[trained_features]

    # Predict user scores
    user_scores = xgb_classifier.predict(user_features)
    recommended_indices = np.argsort(user_scores)[::-1]

    # Filter recommended products based on user address
    if user_address:
        recommended_products = product_data.iloc[recommended_indices]
        recommended_products = recommended_products[recommended_products['product_availability'] == user_address]

        # Fallback: If no products are available in the user's region, recommend globally popular products
        if recommended_products.empty:
            print(f"No products available in {user_address}. Recommending globally popular products.")
            return popularity_based_recommendation(top_n=top_n)

        return recommended_products.head(top_n)[['product_id', 'rating']]
    else:
        return product_data.iloc[recommended_indices[:top_n]][['product_id', 'rating']]

def hybrid_recommendation(user_id=None, product_id=None, user_address=None, is_new_user=False, top_n=20, is_new_product=False):
    if is_new_user:
        return popularity_based_recommendation(user_address, top_n)
    elif is_new_product:
        return content_based_recommendation(product_id, user_address, top_n)
    elif product_id:
        return content_based_recommendation(product_id, user_address, top_n)
    elif user_id:
        return collaborative_filtering_recommendation(user_id, user_address, top_n)
    return "Invalid input"

# Example Scenarios
# print("\nRecommendations for New User in New York:")
# print(hybrid_recommendation(is_new_user=True, user_address='Andhra Pradesh', top_n=20))

#  done print("\nRecommendations for Existing User in Los Angeles with Product Clicked:")
# print(hybrid_recommendation(user_id='user_17', product_id='prod_110', user_address='Andhra Pradesh', top_n=20))

# print("\nRecommendations for Existing User in Chicago with Historical Data:")
# print(hybrid_recommendation(user_id='user_17', user_address='Andhra Pradesh', is_new_user=False, top_n=20))

# print("\nRecommendations for New Product in Houston:")
# print(hybrid_recommendation(product_id='prod_117', user_address='Jharkhand', is_new_product=True, top_n=20))



# views.py

from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth import hashers
from .models import Profile
from django.shortcuts import get_object_or_404

def otp_generator():
    return  ''.join(random.choices('0123456789', k=6))


    
@csrf_exempt
def create_user(request):
    if request.method == 'POST':
        # Get the data from the POST request
        username = request.POST['username']
        password1 = request.POST['password1']
        password2 = request.POST['password2']
        email = request.POST['email']
        first_name = request.POST['firstname']
        last_name = request.POST['lastname']
        phone_number = request.POST['phone_number']
        role = request.POST['role']  # Get role from dropdown
        state = get_object_or_404(State, id=request.POST['state'])
        district = get_object_or_404(District, id=request.POST['district'])
        subdistrict = get_object_or_404(SubDistrict,id=request.POST['subdistrict'])
        village = get_object_or_404(Village, id=request.POST['village'])
        try:
            validate_email(email)
        except ValidationError:
            return render(request, 'blog.html', {'error': 'Invalid email format!'})

        # Validate passwords match
        if password1 != password2:
            return render(request, 'index.html', {'error_message': 'Passwords do not match'})

        # Check if username or email already exists
        if User.objects.filter(username=username).exists():
            return render(request, 'index.html', {'error_message': 'Username already exists'})
        if User.objects.filter(email=email).exists():
            return render(request, 'index.html', {'error_message': 'Email already registered'})

        # Generate OTP
        otp = otp_generator()
        # Send OTP to email
        send_mail(
            'Confirm Account',
            f'Your OTP for confirming your account is: {otp}',
            settings.EMAIL_HOST_USER,
            [email],
            fail_silently=False,
        )

        # Render OTP validation page
        return render(request, 'otp_validation.html', {
            'username': username,
            'password': password1,
            'email': email,
            'first_name': first_name,
            'last_name': last_name,
            'state': state,
            'district': district,
            'subdistrict': subdistrict,
            'village': village,
            'phone_number': phone_number,
            'role': role,
            'otp': otp,
        })

    return render(request, 'index.html')



from django.contrib.auth import authenticate, login

def verify_otp(request):
        
    if request.method == 'POST':
       
        otp_entered = request.POST.get('otp_entered')
        otp_sent = request.POST.get('otp_sent')  # OTP sent from the create_user view

        # Verify OTP
        if otp_entered == otp_sent:
            # Create the user and profile
             # Get the data from the POST request
            username = request.POST.get('username')
            password = request.POST.get('password')
            email = request.POST.get('email')
            first_name = request.POST.get('first_name')
            last_name = request.POST.get('last_name')
            state = request.POST.get('state')
            district = request.POST.get('district')
            subdistrict = request.POST.get('subdistrict')
            village = request.POST.get('village')
            phone_number = request.POST.get('phone_number')
            role = request.POST.get('role')
            user = User.objects.create_user(
                username=username,
                email=email,
                password=password,
                first_name=first_name,
                last_name=last_name,
            )
            user.save()

            profile = Profile.objects.create(
                user=user,
                state=state,
                district=district,
                subdistrict=subdistrict,
                village=village,
                phone_number=phone_number,
                role=role,
            )
            profile.save()

            # Send confirmation email
            send_mail(
                'Account Created Successfully',
                f"Hello {first_name} {last_name}, your account has been created successfully!",
                settings.EMAIL_HOST_USER,
                [email],
                fail_silently=False,
            )

            # Log the user in
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)

            type_of_profile=role
           

           

            if type_of_profile=='seller':
                 context = {'user': user,'profile': user.profile if hasattr(user, 'profile') else None}
                 return render(request, 'seller_profile_page.html', {'context':context})
            if type_of_profile=='customer':
                 # Display all products posted by all sellers
                seller_profiles = Profile.objects.filter(role="seller").values_list('user', flat=True)
                products = Product.objects.filter(seller__id__in=seller_profiles)
                wishlist_product_ids = Wishlist.objects.filter(customer=request.user).values_list('product_id', flat=True)
                print(list(wishlist_product_ids))
                context={'products':products,'user':user,'profile':user.profile if hasattr(user, 'profile') else None,'wishlist_product_ids': list(wishlist_product_ids)}
                return render(request, "customer_profile_page.html", context)




        else:
            # Invalid OTP
            return render(request, 'otp_validation.html', {'email': email, 'error_message': 'Invalid OTP'})

    return redirect('index')


def send_otp_by_forgot_password(request):
        if request.method == "POST":
            flag = request.POST.get('otp_sent_for_forgot_password')
            if flag:
                
                otp_entered = request.POST.get('otp_entered')
                otp_sent = request.POST.get('otp_sent')
                email = request.POST.get('email')
                if otp_entered == otp_sent:
                    return render(request,'reset_password.html',{'email':email})
        
            else:
                email = request.POST.get('email')
                if  email and User.objects.filter(email=email).exists():
                    otp = otp_generator()
                    # Send OTP to email
                    send_mail(
                        'Confirm Account',
                        f'Your OTP for confirming your account is: {otp}',
                        settings.EMAIL_HOST_USER,
                        [email],
                        fail_silently=False,
                    )
                    return render(request,'otp_validation.html',{'otp':otp,'forgot_password_flag':True,'email':email})
                else:
                    messages.error(request, "Please enter a valid email address.")
                    return redirect('index')  # Replace 'submit_email' with the appropriate URL name
                
def change_password(request):
    if request.method == "POST":
        email = request.POST.get("email")
        new_password = request.POST.get("new_password")
        confirm_password = request.POST.get("confirm_password")

        # Check if all fields are provided
        if not email or not new_password or not confirm_password:
            return render(request, "reset_password.html", {"error": "All fields are required", "email": email})

        # Check if passwords match
        if new_password != confirm_password:
            return render(request, "reset_password.html", {"error": "Passwords do not match", "email": email})

        # Verify if user exists
        try:
            user = User.objects.get(email=email)
        except User.DoesNotExist:
            return render(request, "reset_password.html", {"error": "User does not exist", "email": email})

        # Update password
        user.password = make_password(new_password)
        user.save()

        return render(request, "reset_password.html", {"success": "Password reset successfully!"})

    return render(request, "reset_password.html")
            





from datetime import timedelta
from django.http import HttpResponse
from urllib.parse import unquote




@csrf_exempt
def login_view(request):
    if request.method == "POST":
        # Get form data
        email = request.POST.get("email")  # Assuming the email is passed in the "username" field
        password = request.POST.get("password")
        role = request.POST.get("role")
        print('inside login')
        
        # Validate form data
        if not email or not password or not role:
            print('eeror in if')
            messages.error(request, "All fields are required.")
            return redirect("index")  # Replace with your login page URL name
        

        

        print('after validation')

        # Check if a user with the given email exists
        

        # Authenticate user
        
        user = authenticate(request, username=email, password=password)
        if user is None:
            print('user is none')

        if user is not None:
            # Additional role check (if roles are stored in user profile or model)
            if hasattr(user, "profile") and user.profile.role != role:
                messages.error(request, "Role does not match.")
                return redirect("index")  # Replace with your login page URL name

            # Log in the user
            
            login(request, user)
            
            if role == "customer":
                customer = request.user  # Logged-in customer

                # Get the customer's profile
                customer_profile = Profile.objects.filter(user=customer).select_related('user').first()

                if not customer_profile:
                    return HttpResponse("Profile not found. Please complete your profile.", status=400)

                user_address = customer_profile.state  # Use safely after profile existence check

                # Check if the user joined more than 10 days ago
                is_new_user = True  # Default to new user
                if request.user.date_joined:  # Ensure date_joined exists
                    is_new_user = (now() - request.user.date_joined).days <= 10

                # Call hybrid recommendation system
                user_id = f'user_{request.user.id}' if request.user.is_authenticated else None
                recommendations = hybrid_recommendation(user_id=user_id, user_address=user_address, is_new_user=is_new_user, top_n=20)
                # ✅ Ensure recommendations is a DataFrame and Extract IDs
                recommendation_ids = []
                if isinstance(recommendations, pd.DataFrame) and 'product_id' in recommendations.columns:
                    recommendations['product_id'] = recommendations['product_id'].astype(str).str.extract(r'prod_(\d+)')[0]
                    recommendation_ids = recommendations['product_id'].dropna().astype(int).tolist()

                # ✅ Fetch recommended products
                products = Product.objects.filter(id__in=recommendation_ids)
                
                print("\nRecommendations:", recommendations)
                print(request.user.id)

                # # Get seller profiles in the same state
                # seller_profiles = Profile.objects.filter(role="seller", state=user_address).values_list('user', flat=True)

                # # Fetch products only from sellers in the same state
                # products = Product.objects.filter(seller__id__in=seller_profiles)

                # Fix image URLs
                # Fix image URLs
                from urllib.parse import unquote

                fixed_rec_images = []
                for rec_product in products:
                    rec_images = []
                    for img in rec_product.images.all():
                        image_url = img.image.url
                        fixed_image_url = unquote(image_url[7:]) if image_url.startswith("/media/https%3A") else image_url
                        rec_images.append(fixed_image_url)

                    # Use default image if no images exist
                    if not rec_images:
                        rec_images.append("https://upload.wikimedia.org/wikipedia/commons/8/8a/Banana-Single.jpg")

                    fixed_rec_images.append(rec_images)

                recommended_products_data = list(zip(products, fixed_rec_images))



                # Fetch wishlist product IDs for the logged-in customer
                wishlist_product_ids = Wishlist.objects.filter(customer=customer).values_list('product_id', flat=True)

                context = {
                    'products': products,
                    'user': customer,
                    'profile': customer_profile,
                    'wishlist_product_ids': list(wishlist_product_ids),
                    'recommended_products_data': recommended_products_data
                }

                return render(request, "customer_profile_page.html", context)
                
            elif role == "seller":
                from urllib.parse import unquote

                # Get the logged-in seller
                seller = request.user  

                # Fetch seller's products efficiently
                seller_products = Product.objects.filter(seller=seller).prefetch_related('images')

                # Fetch orders that contain the seller's products
                order_items = OrderItem.objects.filter(product__in=seller_products).select_related('order', 'product')

                # Get the most recent 3 distinct orders
                orders = Order.objects.filter(items__in=order_items).distinct().order_by('-placed_at')[:3]

                # Process product images
                for product in seller_products:
                    first_image = product.images.first()  # Get first image if exists
                    if first_image:
                        image_url = first_image.image.url  
                        if image_url.startswith("/media/https%3A"):  
                            product.fixed_image_url = unquote(image_url[7:])  # Decode external URL
                        else:
                            product.fixed_image_url = image_url  # Use local image
                    else:
                        product.fixed_image_url = "https://upload.wikimedia.org/wikipedia/commons/8/8a/Banana-Single.jpg"  # Default image

                # Ensure the seller has a profile before accessing it
                seller_profile = getattr(seller, 'profile', None)

                context = {
                    'user': seller,
                    'profile': seller_profile,  # Avoids errors if profile doesn't exist
                    'products': seller_products,  # Now includes `fixed_image_url`
                    'orders': orders,
                }

                return render(request, "seller_profile_page.html", context)  # Redirect to seller dashboard
        else:
            messages.error(request, "Invalid email or password.")
            return redirect("index")  # Replace with your login page URL name
    else:
        # Render the login page for GET requests
        return render(request, "index.html")
def customer_home_page(request):
    user=request.user
    # Display all products posted by all sellers
    seller_profiles = Profile.objects.filter(role="seller").values_list('user', flat=True)
    products = Product.objects.filter(seller__id__in=seller_profiles)
    from urllib.parse import unquote
    for product in products:
        first_image = product.images.first()
        if first_image:
            image_url = first_image.image.url  # Local image
            if image_url.startswith("/media/https%3A"):  
                product.fixed_image_url = unquote(image_url[7:])  # Decode URL
            else:
                product.fixed_image_url = image_url
        else:
            product.fixed_image_url = "https://upload.wikimedia.org/wikipedia/commons/8/8a/Banana-Single.jpg"
    wishlist_product_ids = Wishlist.objects.filter(customer=request.user).values_list('product_id', flat=True)
    print(list(wishlist_product_ids))
    context={'products':products,'user':user,'profile':user.profile if hasattr(user, 'profile') else None,'wishlist_product_ids': list(wishlist_product_ids)}
    return render(request, "customer_profile_page.html", context)

def forgot_password(request):
    return render(request,'send_otp.html')
    





def logout_view(request):
    """
    Logs out the currently authenticated user and redirects to the homepage.
    """
    if request.user.is_authenticated:
        logout(request)
        messages.success(request, "You have been logged out successfully.")
    else:
        messages.info(request, "You are not logged in.")
    return render(request,'index.html')  # Replace 'index' with your homepage URL name



from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from .models import Product, ProductImage

@login_required
@csrf_exempt
def add_product(request):
    if request.method == 'POST':
        # Extract product details from the POST request
        name = request.POST.get('productName')
        category = request.POST.get('productCategory')
        unit = request.POST.get('productunit')
        price_per_unit = request.POST.get('productPrice')
        quantity_in_stock = request.POST.get('productQuantity')
        description = request.POST.get('productDescription')

        # Validate required fields
        if not all([name, category, unit, price_per_unit, quantity_in_stock, description]):
            return JsonResponse({'error': 'All fields are required.'}, status=400)
        
        try:
            price_per_unit = float(price_per_unit)
            quantity_in_stock = int(quantity_in_stock)
        except ValueError:
            return JsonResponse({'error': 'Invalid price or quantity format.'}, status=400)

        # Create a new product
        product = Product.objects.create(
            seller=request.user,  # Assuming the logged-in user is the seller
            name=name,
            description=description,
            price_per_unit=price_per_unit,
            quantity_in_stock=quantity_in_stock,
            category=category,
            unit=unit,
        )

        # Handle uploaded images
        images = request.FILES.getlist('productImages[]')  # Retrieve multiple images
        for image in images:
            ProductImage.objects.create(product=product, image=image)
    
    # Fetch products that belong to the logged-in user (the seller)
    user_products = Product.objects.filter(seller=request.user)

   

        
    user=request.user
    context = {
                'user': user,
                'profile': user.profile if hasattr(user, 'profile') else None,
                'products': user_products
            }

    return render(request, 'seller_profile_page.html',context)  # Template for product addition (if required)


def delete_product(request, product_id):
    # Get the product object and ensure it's the logged-in user's product
    product = get_object_or_404(Product, id=product_id, seller=request.user)
    
    # Delete the product
    product.delete()

    # Redirect to the profile page after deletion
    return redirect('add_product')

@login_required
@csrf_exempt
def edit_product(request):
    if request.method == 'POST':
        product_id = request.POST.get('product_id')
        price_per_unit = request.POST.get('price_per_unit')

        # Ensure the product belongs to the logged-in user
        product = get_object_or_404(Product, id=product_id, seller=request.user)

        # Update the product's price
        product.price_per_unit = price_per_unit
        product.save()

        # Redirect to the user's profile page
        return redirect('add_product')
    
@login_required
@csrf_exempt
def edit_profile(request):
    profile = Profile.objects.get(user=request.user)  # Get the user's profile
    if request.method == 'POST':
        # Directly access POST data
        user = request.user
        new_email = request.POST.get('email', user.email)

        user.email=new_email
        user.first_name=request.POST.get('first_name')
        user.last_name=request.POST.get('last_name')
        user.save()
        profile.phone_number = request.POST.get('phone_number', profile.phone_number)
       
        
        # Save updated profile
        profile.save()
        messages.success(request, 'Profile updated successfully!')
        return redirect('add_product')
    return render(request, 'edit_profile.html', {'profile': profile})



@login_required
@csrf_exempt
def change_password(request):
    if request.method == 'POST':
        
        new_password = request.POST.get('new_password')
        confirm_password = request.POST.get('confirm_password')

        user = request.user

        

        # Check if the new passwords match
        if new_password != confirm_password:
            messages.error(request, 'New passwords do not match.')
            return redirect('change_password')

        # Update the password
        user.set_password(new_password)
        user.save()

        # Update session to prevent logout
        update_session_auth_hash(request, user)
        messages.success(request, 'Password changed successfully!')
        return redirect('add_product')
    
    return render(request, 'change_password.html')



@login_required
def wishlist_toggle(request):
    if request.method == 'POST':
        product_id = request.POST.get('product_id')
        product = get_object_or_404(Product, id=product_id)
        wishlist_item, created = Wishlist.objects.get_or_create(customer=request.user, product=product)
        page_type=request.POST.get('origin')

        # Toggle the wishlist item
        if not created:
            wishlist_item.delete()
            status = 'removed'
        else:
            status = 'added'

        # Update the context for rendering the customer profile page
        seller_profiles = Profile.objects.filter(role="seller").values_list('user', flat=True)
        products = Product.objects.filter(seller__id__in=seller_profiles)
        wishlist_product_ids = Wishlist.objects.filter(customer=request.user).values_list('product_id', flat=True)
        context = {
            'products': products,
            'user': request.user,
            'profile': request.user.profile if hasattr(request.user, 'profile') else None,
            'wishlist_product_ids': list(wishlist_product_ids),
        }

        if page_type:
            return render(request,"wishlist.html",context)

        # Render the same page with updated context
        return render(request, "customer_profile_page.html", context)

    # Fallback in case of a GET request
    return JsonResponse({'error': 'Invalid request method'}, status=400)

from urllib.parse import unquote

def wishlist(request):
    seller_profiles = Profile.objects.filter(role="seller").values_list('user', flat=True)
    products = Product.objects.filter(seller__id__in=seller_profiles)
    wishlist_product_ids = Wishlist.objects.filter(customer=request.user).values_list('product_id', flat=True)

    for product in products:
        first_image = product.images.first()
        if first_image:
            image_url = first_image.image.url  
            if image_url.startswith("/media/https%3A"):  
                product.fixed_image_url = unquote(image_url[7:])  
            else:
                product.fixed_image_url = image_url
        else:
            product.fixed_image_url = "https://via.placeholder.com/250"

    context = {
        'products': products,
        'user': request.user,
        'profile': getattr(request.user, 'profile', None),
        'wishlist_product_ids': list(wishlist_product_ids),
    }
    return render(request, 'wishlist.html', context)







@login_required
def add_to_cart(request, product_id):
    product = get_object_or_404(Product, id=product_id)
    cart_item, created = AddToCart.objects.get_or_create(
        customer=request.user,
        product=product,
    )

    print('entered into cart')

    if not created:
        cart_item.quantity += 1
        cart_item.save()

    
    # Update the context for rendering the customer profile page
    seller_profiles = Profile.objects.filter(role="seller").values_list('user', flat=True)
    products = Product.objects.filter(seller__id__in=seller_profiles)
    wishlist_product_ids = Wishlist.objects.filter(customer=request.user).values_list('product_id', flat=True)
    
    context = {
        'products': products,
        'user': request.user,
        'profile': request.user.profile if hasattr(request.user, 'profile') else None,
        'wishlist_product_ids': list(wishlist_product_ids),
    }

    # Render the same page with updated context
    return render(request, "customer_profile_page.html", context)


from django.shortcuts import render
from urllib.parse import unquote
from django.contrib.auth.decorators import login_required
from .models import AddToCart, ShippingAddress, Profile

@login_required
def cart(request):
    user = request.user
    shipping_address = ShippingAddress.objects.filter(user=user).first()  
    profile = Profile.objects.filter(user=user).first()  

    if shipping_address:
        address = {
            'address_line1': shipping_address.address_line1 or '',
            'address_line2': shipping_address.address_line2 or '',
            'state': shipping_address.state,
            'district': shipping_address.district,
            'subdistrict': shipping_address.subdistrict,
            'village': shipping_address.village,
            'phone_number': shipping_address.phone_number,
            'postal_code': shipping_address.postal_code or '',
            'full_address': f"{shipping_address.address_line1 or ''}, {shipping_address.address_line2 or ''}, "
                            f"{shipping_address.village}, {shipping_address.subdistrict}, {shipping_address.district}, "
                            f"{shipping_address.state}, {shipping_address.postal_code or ''}".strip(', ')
        }
    elif profile:
        address = {
            'state': profile.state,
            'district': profile.district,
            'subdistrict': profile.subdistrict,
            'village': profile.village,
            'full_address': f"{profile.village}, {profile.subdistrict}, {profile.district}, {profile.state}"
        }
    else:
        address = None  

    # Fetch all items for the logged-in user's cart
    cart_items = AddToCart.objects.filter(customer=request.user)
    total_price = sum(item.quantity * item.product.price_per_unit for item in cart_items)

    for item in cart_items:
        item.total_price = item.quantity * item.product.price_per_unit
        
        # Fix Image URL Encoding
        first_image = item.product.images.first()
        if first_image:
            image_url = first_image.image.url
            if image_url.startswith("/media/https%3A"):
                item.fixed_image_url = unquote(image_url[7:])
            else:
                item.fixed_image_url = image_url
        else:
            item.fixed_image_url = "https://via.placeholder.com/250"  # Default image

    return render(request, 'cart.html', {
        'cart_items': cart_items,
        'total_price': total_price,
        'address': address,
    })



from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .models import ShippingAddress

@login_required
def update_address(request):
    if request.method == "POST":
        user = request.user
        shipping_address, created = ShippingAddress.objects.get_or_create(user=user)

        # Update the address fields
        shipping_address.phone_number = request.POST.get("phone_number")
        shipping_address.address_line1 = request.POST.get("address_line1")
        shipping_address.address_line2 = request.POST.get("address_line2")
        shipping_address.state = State.objects.get(id=request.POST.get("state")).name
        shipping_address.district = District.objects.get(id=request.POST.get("district")).name
        shipping_address.subdistrict = SubDistrict.objects.get(id=request.POST.get("subdistrict")).name
        shipping_address.village = Village.objects.get(id=request.POST.get("village")).name
        shipping_address.postal_code = request.POST.get("postal_code")
        shipping_address.save()
        messages.success(request, "Address updated successfully.")
        return redirect("cart")

    return redirect("cart")



@login_required
def update_cart_quantity(request):
    if request.method == 'POST':
        cart_item_id = request.POST.get('cart_item_id')
        action = request.POST.get('quantity_action')
        cart_item = get_object_or_404(AddToCart, id=cart_item_id, customer=request.user)

        if action == 'increase':
            cart_item.quantity += 1
        elif action == 'decrease' and cart_item.quantity > 1:
            cart_item.quantity -= 1

        cart_item.save()
    return redirect('cart')


@login_required
def remove_from_cart(request):
    if request.method == 'POST':
        cart_item_id = request.POST.get('cart_item_id')
        cart_item = get_object_or_404(AddToCart, id=cart_item_id, customer=request.user)
        cart_item.delete()
    return redirect('cart')

from django.shortcuts import get_object_or_404, render
from .models import Product

from urllib.parse import unquote
from django.shortcuts import get_object_or_404, render

from django.shortcuts import render, get_object_or_404
from urllib.parse import unquote

from django.shortcuts import render, get_object_or_404
from urllib.parse import unquote
import pandas as pd

def quick_view(request, product_id):
    product = get_object_or_404(Product, id=product_id)

    # ✅ Fix image URLs for the main product
    fixed_product_images = []
    for img in product.images.all():
        image_url = img.image.url
        fixed_image_url = unquote(image_url[7:]) if image_url.startswith("/media/https%3A") else image_url
        fixed_product_images.append(fixed_image_url)

    # Use default image if no images exist
    if not fixed_product_images:
        fixed_product_images.append("https://upload.wikimedia.org/wikipedia/commons/8/8a/Banana-Single.jpg")

    reviews = product.reviews.all().order_by('-created_at')

    # Hybrid Recommendation for the user who clicked on the product
    user_id = f'user_{request.user.id}' if request.user.is_authenticated else None
    user_address = request.user.profile.state if request.user.is_authenticated else ''

    recommendations = hybrid_recommendation(
        user_id=user_id,
        product_id=f'prod_{product_id}',
        user_address=user_address,
        top_n=10
    )

    # ✅ Ensure recommendations is a DataFrame and Extract IDs
    recommendation_ids = []
    if isinstance(recommendations, pd.DataFrame) and 'product_id' in recommendations.columns:
        recommendations['product_id'] = recommendations['product_id'].astype(str).str.extract(r'prod_(\d+)')[0]
        recommendation_ids = recommendations['product_id'].dropna().astype(int).tolist()

    # ✅ Fetch recommended products
    recommended_products = Product.objects.filter(id__in=recommendation_ids)

    # ✅ Fix image URLs for recommended products
    fixed_rec_images = []
    for rec_product in recommended_products:
        rec_images = []
        for img in rec_product.images.all():
            image_url = img.image.url
            fixed_image_url = unquote(image_url[7:]) if image_url.startswith("/media/https%3A") else image_url
            rec_images.append(fixed_image_url)

        # Use default image if no images exist
        if not rec_images:
            rec_images.append("https://upload.wikimedia.org/wikipedia/commons/8/8a/Banana-Single.jpg")

        fixed_rec_images.append(rec_images)

    recommended_products_data = list(zip(recommended_products, fixed_rec_images))

    return render(request, 'product.html', {
        'product': product,
        'product_images': fixed_product_images,  # Main product images
        'reviews': reviews,
        'recommended_products_data': recommended_products_data  # List of tuples (Product, Fixed Images)
    })





from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .models import AddToCart, Order, OrderItem, ShippingAddress

from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .models import AddToCart, Order, OrderItem, ShippingAddress

from django.shortcuts import redirect
from django.contrib.auth.decorators import login_required
from .models import Order, AddToCart, ShippingAddress, OrderItem

@login_required
def place_order(request):
    customer = request.user  # Get the logged-in user

    # Fetch cart items
    cart_items = AddToCart.objects.filter(customer=customer)
    if not cart_items.exists():
        return redirect("cart")  # Redirect if the cart is empty

    # Get the latest shipping address of the user
    try:
        shipping_address = ShippingAddress.objects.filter(user=customer).latest("id")
    except ShippingAddress.DoesNotExist:
        return redirect("shipping_address")  # Redirect if no shipping address found

    # Calculate total price
    total_price = sum(item.product.price_per_unit * item.quantity for item in cart_items)

    # Format shipping details as a single string
    shipping_details = (
        f"{shipping_address.phone_number}\n"
        f"{shipping_address.address_line1}\n"
        f"{shipping_address.address_line2}\n"
        f"{shipping_address.village}\n"
        f"{shipping_address.subdistrict}\n"
        f"{shipping_address.district}\n"
        f"{shipping_address.state}\n"
        f"{shipping_address.postal_code}"
    ).strip()

    # Create an Order with shipping details in a single field
    order = Order.objects.create(
        customer=customer,
        shipping_details=shipping_details,
        total_price=total_price,
    )

    # Create OrderItems from cart
    order_items = [
        OrderItem(order=order, product=item.product, quantity=item.quantity, price_at_purchase=item.product.price_per_unit)
        for item in cart_items
    ]
    OrderItem.objects.bulk_create(order_items)  # Save all order items at once

    # Clear the cart
    cart_items.delete()

    return redirect("cart")  # Redirect to an order success page





from django.http import JsonResponse
from django.views.decorators.http import require_GET
from .models import State, District, SubDistrict, Village

@require_GET
def get_states(request):
    states = State.objects.only('id', 'name')
    return JsonResponse(list(states.values('id', 'name')), safe=False)

@require_GET
def get_districts(request):
    state_id = request.GET.get('state_id')
    if not state_id:
        return JsonResponse({"error": "Missing state_id"}, status=400)
    
    districts = District.objects.filter(state_id=state_id).only('id', 'name')
    return JsonResponse(list(districts.values('id', 'name')), safe=False)

@require_GET
def get_subdistricts(request):
    district_id = request.GET.get('district_id')
    if not district_id:
        return JsonResponse({"error": "Missing district_id"}, status=400)

    subdistricts = SubDistrict.objects.filter(district_id=district_id).only('id', 'name')
    return JsonResponse(list(subdistricts.values('id', 'name')), safe=False)

@require_GET
def get_villages(request):
    subdistrict_id = request.GET.get('subdistrict_id')
    if not subdistrict_id:
        return JsonResponse({"error": "Missing subdistrict_id"}, status=400)

    villages = Village.objects.filter(subdistrict_id=subdistrict_id).only('id', 'name')
    return JsonResponse(list(villages.values('id', 'name')), safe=False)
from django.shortcuts import render, get_object_or_404
from django.utils.timezone import now
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
import json
from .models import Order, OrderItem, ProductReview, Product

@login_required
def orders(request):
    orders = Order.objects.filter(customer=request.user).order_by("-placed_at").prefetch_related("items__product__images")
    for order in orders:
        time_diff = (now() - order.placed_at).total_seconds()

        if time_diff >= 60:
            order.status = "delivered"
        elif time_diff >= 30:
            order.status = "out_for_delivery"
        elif time_diff >= 15:
            order.status = "shipped"
        else:
            order.status = "ordered"

        order.save(update_fields=["status"])

        
        # Attach product reviews and modify image URLs
        for item in order.items.all():
            item.review = ProductReview.objects.filter(product=item.product, customer=request.user).first()
            
            first_image = item.product.images.first()
            if first_image:
                image_url = first_image.image.url  # Default local image

                # Fix the image URL if it contains encoded external links
                if image_url.startswith("/media/https%3A"):  
                    item.fixed_image_url = unquote(image_url[7:])  # Decode URL
                else:
                    item.fixed_image_url = image_url  # Keep local URL
            else:
                # Set a fallback image if no image exists
                item.fixed_image_url = "https://upload.wikimedia.org/wikipedia/commons/8/8a/Banana-Single.jpg"


    return render(request, "orders.html", {"orders": orders})

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from .models import Product, ProductReview

@login_required
def submit_review(request):
    if request.method == "POST":
        product_id = request.POST.get("product_id")
        rating = request.POST.get("rating")
        comment = request.POST.get("comment")
        

        # Validate the input
        if not product_id or not rating or not comment:
            
            return redirect("orders")  # Redirect back if invalid

        product = get_object_or_404(Product, id=product_id)
        rating = int(rating)  # Convert rating to integer

        # Check if user has already reviewed the product
        review, created = ProductReview.objects.get_or_create(
            product=product,
            customer=request.user,
            defaults={"rating": rating, "comment": comment},
        )

        if not created:
            # Update existing review
            review.rating = rating
            review.comment = comment
            review.save()

        return redirect("orders")  # Redirect to product page

    return redirect("orders")  # Redirect to home if not POST request




# Create your views here.
from django.shortcuts import render
from .models import Product, ProductReview



def index(request):
    reviews = ProductReview.objects.select_related('customer').order_by('-created_at')[:5]  # Fetch latest 5 reviews
    return render(request, 'index.html', {'reviews': reviews})


from django.shortcuts import render
from urllib.parse import unquote
from .models import Product, Profile, Wishlist
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from urllib.parse import unquote
from .models import Product, Profile, Wishlist, State

@csrf_exempt
def shop(request):
    customer = request.user  # Logged-in customer

    # Get filtering parameters from request (use POST instead of GET)
    selected_state_id = request.POST.get('state', '')  # State ID, not name
    selected_category = request.POST.get('category', '')  # Category name
    max_price = request.POST.get('max_price', '')  # Max price input

    # Convert max_price to integer if provided
    try:
        max_price = int(max_price) if max_price.isdigit() else None
    except ValueError:
        max_price = None  # Reset if invalid input

    # Get state name from ID
    selected_state_name = None
    if selected_state_id:
        state_obj = State.objects.filter(id=selected_state_id).first()
        if state_obj:
            selected_state_name = state_obj.name  # Get state name from model

    # Get sellers based on selected state (if any)
    seller_profiles = Profile.objects.filter(role="seller")
    if selected_state_name:
        seller_profiles = seller_profiles.filter(state=selected_state_name)  # Use name, not ID

    seller_ids = seller_profiles.values_list('user', flat=True)

    # Fetch products from sellers with filters
    products = Product.objects.filter(seller__id__in=seller_ids, price_per_unit__gte=10)  # Min price always 10

    # Apply max price filter if provided
    if max_price is not None:
        products = products.filter(price_per_unit__lte=max_price)

    # Apply category filter only if a category is selected
    if selected_category and selected_category.lower() != "no selection":
        products = products.filter(category=selected_category)

    # Limit to 500 products after filtering
    products = products[:500]

    # Fix image URLs
    default_image = "https://upload.wikimedia.org/wikipedia/commons/8/8a/Banana-Single.jpg"
    for product in products:
        first_image = product.images.first()
        if first_image:
            image_url = first_image.image.url
            product.fixed_image_url = unquote(image_url[7:]) if image_url.startswith("/media/https%3A") else image_url
        else:
            product.fixed_image_url = default_image

    # Fetch wishlist product IDs for the logged-in customer
    wishlist_product_ids = set(Wishlist.objects.filter(customer=customer).values_list('product_id', flat=True))

    context = {
        'products': products,
        'user': customer,
        'wishlist_product_ids': wishlist_product_ids,
        'selected_state': selected_state_id,  # Keep state ID in form for re-selection
        'selected_category': selected_category,
        'max_price': max_price if max_price is not None else '',
    }

    return render(request, "shop.html", context)


    


def about(request):
    return render(request,'about.html')



def blog(request):
    return render(request,'blog.html')

def contact(request):
    return render(request,'contact.html')



from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from .models import ChatRoom, Message, Product
from django.contrib.auth.models import User

@login_required
def create_or_redirect_chat(request):
    """Creates a chat room if not exists or redirects to an existing one."""
    if request.method == "POST":
        product_id = request.POST.get("product_id")
        product = get_object_or_404(Product, id=product_id)
        farmer = product.seller
        customer = request.user  

        if customer == farmer:
            return redirect("home") 

        # Unique room name using user IDs
        room_name = f"{min(customer.id, farmer.id)}_{max(customer.id, farmer.id)}"

        # Get or create the chat room
        room, created = ChatRoom.objects.get_or_create(name=room_name)
        room.users.add(customer, farmer)

        return redirect("chat_room", room_name=room.name)

    return redirect("home")

from django.shortcuts import render, get_object_or_404
from django.contrib.auth.decorators import login_required
from .models import ChatRoom, Message

from django.shortcuts import render, get_object_or_404
from django.contrib.auth.decorators import login_required
from .models import ChatRoom, Message

from django.shortcuts import render, get_object_or_404
from django.contrib.auth.decorators import login_required
from .models import ChatRoom, Message

@login_required
def chat_room(request, room_name=None):
    current_user = request.user

    # Get all chat rooms where the current user is a participant
    chat_rooms = ChatRoom.objects.prefetch_related("users").filter(users=current_user)

    # Preprocess chat rooms to include the other user's details
    chat_rooms_with_other_user = [
        {"room": chat_room, "other_user": chat_room.users.exclude(id=current_user.id).first()}
        for chat_room in chat_rooms
    ]

    # If no room_name is provided, select the most recent one
    if not room_name and chat_rooms.exists():
        latest_message = Message.objects.filter(room__in=chat_rooms).order_by("-timestamp").first()
        room_name = latest_message.room.name if latest_message else chat_rooms.first().name

    # Fetch the selected chat room, handling cases where no room exists
    room = get_object_or_404(ChatRoom, name=room_name) if room_name else None

    # Fetch the other user in the selected chat room
    other_user = room.users.exclude(id=current_user.id).first() if room else None

    # Fetch messages for the selected chat room
    messages = Message.objects.filter(room=room).order_by("timestamp") if room else []

    return render(request, "chat.html", {
        "chat_rooms_with_other_user": chat_rooms_with_other_user,  # Pass preprocessed chat rooms
        "room": room,
        "other_user": other_user,  # Pass the other user for the selected room
        "messages": messages,
    })



from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from .models import Message
import json

from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
import json
from .models import Message

from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
import json
from .models import Message, ChatRoom

@login_required
def mark_seen(request):
    if request.method == "POST":
        data = json.loads(request.body)
        chat_with = data.get("chat_with")  # Username of the other user

        if chat_with:
            try:
                # Find the chat room where both the current user and the other user are participants
                chat_room = ChatRoom.objects.filter(users=request.user).filter(users__username=chat_with).first()

                if not chat_room:
                    return JsonResponse({"status": "error", "message": "Chat room not found."}, status=404)

                # Filter messages sent by the other user in this chat room that are unread
                messages = Message.objects.filter(
                    sender__username=chat_with,  # Messages sent by the other user
                    room=chat_room,  # Messages in the specific chat room
                    is_seen=False  # Only unread messages
                )

                # Mark messages as seen
                messages.update(is_seen=True)

                return JsonResponse({"status": "success", "message": "Messages marked as seen."})
            except Exception as e:
                return JsonResponse({"status": "error", "message": str(e)}, status=500)
        else:
            return JsonResponse({"status": "error", "message": "Invalid 'chat_with' parameter."}, status=400)
    return JsonResponse({"status": "error", "message": "Invalid request method."}, status=400)


from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from .models import Message, ChatRoom

@login_required
def get_chat_history(request):
    chat_with = request.GET.get("chat_with")
    if not chat_with:
        return JsonResponse({"status": "error", "message": "Invalid 'chat_with' parameter."}, status=400)

    try:
        # Find the chat room where both the current user and the other user are participants
        chat_room = ChatRoom.objects.filter(users=request.user).filter(users__username=chat_with).first()

        if not chat_room:
            return JsonResponse({"status": "error", "message": "Chat room not found."}, status=404)

        # Fetch messages in the chat room
        messages = Message.objects.filter(room=chat_room).order_by("timestamp")

        # Serialize messages
        messages_data = [
            {
                "sender": message.sender.username,
                "content": message.content,
                "timestamp": message.timestamp.isoformat(),
                "is_seen": message.is_seen
            }
            for message in messages
        ]

        return JsonResponse({"status": "success", "messages": messages_data})
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)