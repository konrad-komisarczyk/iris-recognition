# Iris-recognition

Niniejszy projekt stanowi przedmiot pracy inżynierskiej realizowanej na Politechnice Warszawskiej na wydziale Matematyki i Nauk Informacyjnych w roku 2023/2024.

Zawiera rozwiązanie w tematyce rozpoznawania osób na podstawie zdjęcia tęczówki.

## Instrukcja instalacji

* `git clone https://github.com/konrad-komisarczyk/iris-recognition`
* stworzenie nowego wirtualnego środowiska korzystającego z Pythona w wersji
3.10.1  i przejście do katalogu projektu
* dodanie wersji bibliotek `torch` i `torchvision` do `requirements.txt` wykorzystujących CUDA, jeżeli CUDA jest dostępne
* `pip install -r requirements.txt`
* `python install.py`
* aplikacja webowa może być uruchomiona w dowolnym momencie z użyciem polecenia:
`python irisverify/manage.py runserver`

W przypadku niepowodzenia należy skorzystać z dokładnej instrukcji 
wdrożeniowej, która znajduje się w załączniku pracy.



Autorzy: Konrad Komisarczyk, Jędrzej Sokołowski, Hubert Kozubek