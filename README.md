# Proyek Analisis Sentimen Ulasan WhatsApp

## Deskripsi Proyek
Proyek ini bertujuan untuk menganalisis sentimen ulasan pengguna aplikasi WhatsApp di Google Play Store. Ulasan diambil secara otomatis menggunakan library `google-play-scraper`, kemudian diproses dan dianalisis untuk menentukan sentimen (positif, negatif, netral) menggunakan berbagai model machine learning dan deep learning. Model yang digunakan meliputi Logistic Regression, SVM, Random Forest, dan Bi-GRU. Selain itu, proyek ini juga mendukung deteksi otomatis ulasan baru dan prediksi sentimennya.

### Tujuan
- Mengumpulkan ulasan WhatsApp dari Google Play Store.
- Melakukan preprocessing teks ulasan.
- Melatih model untuk mengklasifikasikan sentimen ulasan.
- Membuat fungsi inference untuk memprediksi sentimen teks baru.
- Mendeteksi ulasan baru secara otomatis dan memprediksi sentimennya.

### Dataset
- **Sumber Data**: Ulasan WhatsApp dari Google Play Store (bahasa Indonesia, `lang='id'`).
- **Jumlah Data**: 11.000 ulasan.
- **Label Sentimen**: Positif, Negatif, Netral (berdasarkan skor ulasan dan lexicon-based labeling).

## Struktur Direktori
```
├── [Scraping]Submission_Proyek_Analisis_Sentimen_Richelle Vania Thionanda.ipynb  
├── [Training]Submission_Proyek_Analisis_Sentimen_Richelle Vania Thionanda.ipynb 
├── ulasan_aplikasi.csv                                                   
├── requirements.txt                                                     
└── README.md                                                    
```

## Prasyarat
### Dependensi
Pastikan Anda telah menginstal Python 3.13.2 atau versi yang kompatibel, serta library berikut:

```bash
pip install google-play-scraper pandas numpy nltk scikit-learn torch matplotlib seaborn wordcloud
```

### Unduh Resource NLTK
Jalankan perintah berikut di Python untuk mengunduh resource NLTK yang diperlukan:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### Persiapan Lingkungan
- Pastikan Anda memiliki koneksi internet untuk scraping ulasan dari Google Play Store.
- Jika menggunakan GPU untuk pelatihan Bi-GRU, pastikan CUDA diinstal dan kompatibel dengan PyTorch.

## Langkah-langkah Menjalankan Proyek

### 1. Scraping dan Preprocessing Data
1. Buka notebook `[Scraping]Submission_Proyek_Analisis_Sentimen_Richelle Vania Thionanda.ipynb`.
2. Jalankan sel secara berurutan untuk:
   - Mengimpor library.
   - Mengambil 11.000 ulasan WhatsApp dari Google Play Store.
   - Melakukan EDA (Exploratory Data Analysis) untuk memeriksa missing values dan duplikat.
   - Melakukan preprocessing teks (case folding, tokenisasi, penanganan slang, stopwords removal, dan negasi).
   - Memberi label sentimen berdasarkan skor ulasan dan lexicon.
   - Menyimpan hasil ke file `ulasan_aplikasi.csv`.

### 2. Pelatihan Model
1. Buka notebook `[Training]Submission_Proyek_Analisis_Sentimen_Richelle Vania Thionanda.ipynb`.
2. Jalankan sel secara berurutan untuk:
   - Mengimpor library dan data dari `ulasan_aplikasi.csv`.
   - Melatih model machine learning (Logistic Regression, SVM, Random Forest) dan deep learning (Bi-GRU).
   - Mengevaluasi performa model menggunakan metrik akurasi, precision, recall, dan F1-score.

   **Hasil Performa Model**:
   - Logistic Regression + TF-IDF: Akurasi Testing 91.86%
   - SVM + TF-IDF: Akurasi Testing 93.23%
   - Random Forest + CountVectorizer: Akurasi Testing 89.09%
   - Bi-GRU: Akurasi Testing 96.18% (model terbaik)


### 3. Inference dan Deteksi Otomatis
1. Tambahkan kode berikut di akhir notebook `[Training]Submission_Proyek_Analisis_Sentimen_Richelle Vania Thionanda.ipynb` untuk inference dan deteksi otomatis:

   ```python
   # Fungsi untuk mengambil ulasan baru secara otomatis
   def fetch_new_reviews(app_id, count=5, lang='id', country='id'):
       try:
           result, _ = reviews(
               app_id,
               lang=lang,
               country=country,
               sort=Sort.NEWEST,
               count=count
           )
           reviews_text = [review['content'] for review in result]
           return reviews_text
       except Exception as e:
           print(f"Error saat mengambil ulasan: {e}")
           return []

   # Fungsi preprocessing teks
   def preprocess_text(text, word_index, max_len=50):
       tokens = word_tokenize(text.lower())
       cleaned_tokens = [token.strip(string.punctuation) for token in tokens if token.strip(string.punctuation)]
       sequence = [word_index.get(token, 0) for token in cleaned_tokens]
       if len(sequence) < max_len:
           sequence += [0] * (max_len - len(sequence))
       else:
           sequence = sequence[:max_len]
       return torch.tensor([sequence], dtype=torch.long).to(device)

   # Fungsi prediksi sentimen
   def predict_sentiment(text, model, word_index, label_encoder, max_len=50):
       model.eval()
       with torch.no_grad():
           input_tensor = preprocess_text(text, word_index, max_len)
           output = model(input_tensor)
           _, predicted = torch.max(output, 1)
           sentiment = label_encoder.inverse_transform(predicted.cpu().numpy())[0]
       return sentiment

   # Fungsi untuk deteksi otomatis dan prediksi
   def auto_detect_and_predict(app_id='com.whatsapp', count=5):
       print(f"\n--- Mengambil {count} Ulasan Terbaru dari {app_id} ---")
       new_reviews = fetch_new_reviews(app_id, count=count)
       if not new_reviews:
           print("Tidak ada ulasan yang berhasil diambil.")
           return
       print("\n--- Hasil Prediksi Sentimen untuk Ulasan Terbaru ---")
       for i, review in enumerate(new_reviews, 1):
           predicted_sentiment = predict_sentiment(review, model, word_index, label_encoder)
           print(f"Ulasan {i}: {review}")
           print(f"Prediksi Sentimen: {predicted_sentiment}\n")

   # Jalankan deteksi otomatis
   auto_detect_and_predict(app_id='com.whatsapp', count=5)
   ```

3. Jalankan fungsi `auto_detect_and_predict()` untuk mengambil ulasan baru secara otomatis dan memprediksi sentimennya.
```

## Catatan
- **Koneksi Internet**: Diperlukan untuk scraping ulasan dari Google Play Store.
- **Model Terbaik**: Bi-GRU memberikan akurasi tertinggi (96.18%) dan digunakan untuk inference.
- **Bahasa**: Ulasan diambil dalam bahasa Indonesia (`lang='id'`). Model mungkin perlu penyesuaian untuk bahasa lain.
- **Error Handling**: Jika scraping gagal, kode akan menampilkan pesan error. Pastikan koneksi internet stabil.

## Kontribusi
Proyek ini dibuat oleh Richelle Vania Thionanda sebagai bagian dari tugas analisis sentimen. Jika Anda ingin berkontribusi, silakan fork repositori ini dan buat pull request dengan perbaikan atau fitur baru.

## Lisensi
Proyek ini dilisensikan di bawah [MIT License](LICENSE).

---

File `README.md` ini memberikan gambaran lengkap tentang proyek, langkah-langkah menjalankannya, dan informasi penting lainnya. Anda dapat menyimpan teks di atas ke file `README.md` di direktori proyek Anda.
