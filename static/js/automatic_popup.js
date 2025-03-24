function openPopupAutomatically(id) {
    document.getElementById(id).classList.add('active');
}

setTimeout(function () {
    openPopupAutomatically('login-popup');
}, 1000); // Adjust delay as needed
