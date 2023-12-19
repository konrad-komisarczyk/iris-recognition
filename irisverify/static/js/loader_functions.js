function ShowLoaderLogin() {
    document.getElementById("loader-message").innerText = "Trwa weryfikacja tożsamości użytkownika..."
    document.getElementById("black-overlay").style.display = "block";
}

function ShowLoaderRegister() {
    document.getElementById("loader-message").innerText = "Trwa ekstrakcja cech tęczówki nowego użytkownika..."
    document.getElementById("black-overlay").style.display = "block";
}
