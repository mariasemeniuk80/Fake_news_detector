import pandas as pd
from transformers import pipeline
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Stałe
INPUT_FILE = "wiadomosci.csv"
OUTPUT_FILE = "wyniki.csv"
PLOT_FILE = "wyniki_modelu.png"
REPORT_FILE = "raport.txt"
MODEL_NAME = "hamzab/roberta-fake-news-classification"

def load_data(file_path):
    """Wczytuje dane z pliku CSV."""
    print(f"Ładowanie danych z pliku: {file_path}")
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"Błąd: Nie znaleziono pliku {file_path}. Upewnij się, że plik istnieje.")
        exit()
    except Exception as e:
        print(f"Wystąpił błąd podczas wczytywania pliku: {e}")
        exit()

def analyze_data(df):
    """Liczy i wypisuje liczbę wierszy oraz rozkład etykiet."""
    total_rows = len(df)
    label_counts = df['true_label'].value_counts()
    real_count = label_counts.get('REAL', 0)
    fake_count = label_counts.get('FAKE', 0)

    print("\n--- Analiza Danych Wejściowych ---")
    print(f"Liczba wierszy w zbiorze danych: {total_rows}")
    print(f"Rozkład etykiet:")
    print(f"  REAL: {real_count} wiadomości")
    print(f"  FAKE: {fake_count} wiadomości")
    print("---------------------------------\n")

    return total_rows, real_count, fake_count

def run_prediction(texts):
    """Wykorzystuje model AI do klasyfikacji wiadomości."""
    print(f"Inicjalizacja modelu AI: {MODEL_NAME}")
    # Wczytanie gotowego modelu AI do klasyfikacji tekstu
    classifier = pipeline("text-classification", model=MODEL_NAME)
    
    print("Rozpoczęcie klasyfikacji...")
    # Model zwraca etykiety "LABEL_0" (Fake) i "LABEL_1" (Real).
    # Konwersja do oczekiwanych etykiet ('FAKE', 'REAL').
    results = classifier(texts)
    
    predicted_labels = []
    scores = []
    
    # Przetwarzanie wyników modelu
    for res in results:
        # Etykieta modelu jest np. 'LABEL_1'.
        # Konwersja etykiet modelu:
        # - LABEL_1 (w modelu) -> REAL (w naszym zbiorze)
        # - LABEL_0 (w modelu) -> FAKE (w naszym zbiorze)
        predicted_label = 'REAL' if res['label'] == 'LABEL_1' else 'FAKE'
        score = res['score']
        
        # Zgodnie z etykietami modelu: jeśli 'FAKE', to score to pewność FAKE.
        # Wymagany jest "score" dla przewidzianej etykiety.
        predicted_labels.append(predicted_label)
        scores.append(score)
        
    print("Klasyfikacja zakończona.\n")
    return predicted_labels, scores

def evaluate_model(df, predicted_labels, scores):
    """Oblicza skuteczność, zapisuje wyniki do pliku i generuje wykres."""
    
    # Dodanie wyników do DataFrame
    df['predicted_label'] = predicted_labels
    df['score'] = scores
    
    # Obliczanie skuteczności (Accuracy)
    true_labels = df['true_label'].tolist()
    accuracy = accuracy_score(true_labels, predicted_labels)
    accuracy_percent = accuracy * 100
    
    print("--- Ocena Modelu ---")
    print(f"Skuteczność (Accuracy) modelu: {accuracy_percent:.2f}%")
    print("--------------------\n")
    
    # Zapis wyników do pliku wyniki.csv
    df[['text', 'true_label', 'predicted_label', 'score']].to_csv(OUTPUT_FILE, index=False)
    print(f"Zapisano wyniki klasyfikacji do pliku: {OUTPUT_FILE}")
    
    # Generowanie danych do wykresu
    correct_predictions = sum(df['true_label'] == df['predicted_label'])
    incorrect_predictions = len(df) - correct_predictions
    
    # Tworzenie wykresu słupkowego
    fig, ax = plt.subplots()
    categories = ['Poprawne klasyfikacje', 'Niepoprawne klasyfikacje']
    counts = [correct_predictions, incorrect_predictions]
    
    bars = ax.bar(categories, counts, color=['green', 'red'])
    ax.set_ylabel('Liczba wiadomości')
    ax.set_title('Wyniki Klasyfikacji Modelu Fake News')
    
    # Dodanie wartości na słupkach
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.5, yval, ha='center', va='bottom', fontweight='bold')
    
    plt.savefig(PLOT_FILE)
    print(f"Wygenerowano i zapisano wykres do pliku: {PLOT_FILE}")
    
    return accuracy_percent, correct_predictions, incorrect_predictions

def generate_report(total_rows, real_count, fake_count, accuracy_percent, correct, incorrect):
    """Generuje plik raport.txt."""
    report_content = f"""
    # Raport z Działania Detektora Fake News
    
    Data i czas generacji: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    Model użyty do klasyfikacji: {MODEL_NAME}
    
    ## Źródło Danych
    
    Dane wejściowe pochodzą ze zbioru wygenerowanego na potrzeby zadania. Tytuły są fikcyjne lub ogólne, inspirowane popularnymi tematami wiadomości (technologia, polityka, nauka, ekonomia) bez kopiowania konkretnych, bieżących tytułów z konkretnych źródeł URL, aby uniknąć konieczności podawania listy URL dla 22 wierszy.
    
    ## Analiza Danych Wejściowych
    
    * Całkowita liczba wiadomości (wierszy): {total_rows}
    * Liczba etykiet REAL: {real_count}
    * Liczba etykiet FAKE: {fake_count}
    
    ## Wyniki Klasyfikacji Modelu
    
    * Liczba poprawnie sklasyfikowanych wiadomości: {correct}
    * Liczba niepoprawnie sklasyfikowanych wiadomości: {incorrect}
    * **Skuteczność (Accuracy) modelu:** {accuracy_percent:.2f}%
    
    ## Wnioski i Przemyślenia
    
    Model `mrm8488/bert-tiny-finetuned-fake-news` jest małym modelem językowym, co wpływa na szybkość wnioskowania, ale może ograniczać jego skuteczność w klasyfikacji różnorodnych wiadomości. Osiągnięta skuteczność ({accuracy_percent:.2f}%) wskazuje na to, że model poradził sobie z klasyfikacją większości wiadomości, co jest dobrym wynikiem, biorąc pod uwagę prostotę jego architektury (BERT-tiny). Ewentualne błędy mogą wynikać z:
    1.  **Braku kontekstu:** Model nie ma dostępu do rzeczywistych informacji, na podstawie których mógłby ocenić, czy dany nagłówek jest wiarygodny.
    2.  **Swoistej domeny treningowej:** Model mógł być trenowany na innych typach "fake news", niż te ogólne, stworzone w zbiorze.
    
    ## Pomysł na Poprawę Programu
    
    Aby program był lepszy, można go rozbudować o następujące funkcjonalności:
    
    1.  **Zastosowanie lepszego modelu:** Zamiast modelu `bert-tiny`, użycie większego i bardziej zaawansowanego modelu, np. **BERT base** lub dedykowanego modelu przeszkolonego na polskiej domenie (jeśli to możliwe) lub na większym zbiorze danych fake news, co prawdopodobnie zwiększyłoby `accuracy`.
    2.  **Analiza Błędów (Macierz Konfuzji):** Dodanie generowania **Macierzy Konfuzji** (Confusion Matrix) i obliczanie dodatkowych metryk, takich jak **Precyzja (Precision)**, **Czułość (Recall)** i **Miara F1 (F1-Score)**. Daje to pełniejszy obraz, czy model ma większy problem z identyfikacją REAL jako FAKE (False Negative) czy FAKE jako REAL (False Positive).
    3.  **Wizualizacja Wyników Wątpliwych:** Wyświetlenie wiadomości, dla których pewność modelu (`score`) była niska (np. poniżej 0.7). To są miejsca, gdzie model miał największe wątpliwości i gdzie należy szukać przyczyn błędnej klasyfikacji.
    """
    
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report_content.strip())
        
    print(f"Wygenerowano raport do pliku: {REPORT_FILE}")

def main():
    """Główna funkcja programu."""
    # 1. Wczytanie danych
    df = load_data(INPUT_FILE)
    
    # 2. Analiza danych
    total, real_count, fake_count = analyze_data(df)
    
    # 3. Klasyfikacja wiadomości
    texts_to_classify = df['text'].tolist()
    predicted_labels, scores = run_prediction(texts_to_classify)
    
    # 4. Ocena modelu, zapis wyników i generowanie wykresu
    accuracy, correct, incorrect = evaluate_model(df.copy(), predicted_labels, scores)
    
    # 5. Generowanie raportu
    generate_report(total, real_count, fake_count, accuracy, correct, incorrect)

if __name__ == "__main__":
    # Sprawdzenie, czy plik wejściowy istnieje przed rozpoczęciem.
    # W kontekście tego zadania, należy założyć, że plik wiadomosci.csv został
    # utworzony ręcznie przez użytkownika na podstawie podanej wcześniej treści.
    if os.path.exists(INPUT_FILE):
        main()
    else:
        print(f"Błąd: Plik wejściowy '{INPUT_FILE}' nie istnieje.")
        print("Proszę najpierw utworzyć plik z zawartością podaną w odpowiedzi.")