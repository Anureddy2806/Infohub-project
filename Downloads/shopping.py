import streamlit as st

# App Title
st.title("Buy What U Want With Happyness 🛍️")

# Sample product catalog
products = {
    "Laptop": {"price": 50000, "image": "💻"},
    "Smartphone": {"price": 20000, "image": "📱"},
    "Headphones": {"price": 1500, "image": "🎧"},
    "Shoes": {"price": 3000, "image": "👟"},
    "Backpack": {"price": 1200, "image": "🎒"}
}

# Initialize session state for cart
if "cart" not in st.session_state:
    st.session_state.cart = {}

st.subheader("Explore Our Products")
for product, details in products.items():
    col1, col2 = st.columns([1, 3])
    with col1:
        st.write(details["image"])
    with col2:
        st.write(f"**{product}** - ₹{details['price']}")
        if st.button(f"Add {product} to Cart"):
            if product in st.session_state.cart:
                st.session_state.cart[product] += 1
            else:
                st.session_state.cart[product] = 1

# Display shopping cart
st.subheader("Your Shopping Cart")
if st.session_state.cart:
    total_cost = 0
    for item, quantity in st.session_state.cart.items():
        st.write(f"🛒 {item} (x{quantity}) - ₹{products[item]['price'] * quantity}")
        total_cost += products[item]['price'] * quantity
    st.write(f"**Total Cost: ₹{total_cost}**")
    
    if st.button("Proceed to Checkout"):
        st.success("Thank you for shopping with 'Buy What U Want With Happyness'! 🎉")
else:
    st.write("Your cart is empty. Start shopping and spread happiness!")

# Add a footer message
st.markdown("🌟 *Happiness starts with great shopping!*")