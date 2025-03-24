function openPopupOnClick(id) {
    document.getElementById(id).classList.add('active');
}

function closePopup(id) {
    document.getElementById(id).classList.remove('active');
}

function toggleForm(closeId, openId) {
    closePopup(closeId);
    openPopupOnClick(openId);
}
