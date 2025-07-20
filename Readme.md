# EV Recharging Scheduling System Documentation

## Overview
Sistem ini bertujuan mengalokasikan sesi pengisian ulang pada kendaraan listrik yang sedang melakukan perjalanan jarak jauh, dengan tujuan mengurangi waktu tunggu di SPKLU. Sistem ini menggunakan Genetic Algorithm untuk mengoptimasi pemberian sesi kepada setiap kendaraan listik. 

## Python Version
```bash
Python 3.13.1
```

## Setup

### Clone this repository
Clone repository dengan kode berikut.
```bash
git clone https://github.com/buatanalan/EV-Scheduling
```

### Setup Environmet
1. Buka terminal di root directory
2. Create a virtual environment dengan menjalankan
```bash
python -m venv .venv
```
3. Aktifkan virtual environment dengan menjalankan
```bash
source .venv/bin/activate
```
4. Install seluruh Library yang dibutuhkan dengan menjalankan
```bash
pip install -r requirements.txt
```

### Setup Layanan Pendukung
Terdapat 2 layanan pendukung yang perlu dijalankan di luar program utama. Layanan pendukung tersebut adalah Redis sebagai database dan EQMX sebagai broker MQTT. Langkah-langkah penyetelan kedua layanan tersebut sebagai berikut : 

1. Nyalakan docker
2. Buka Terminal
3. Jalankan redis, atau jika belum memiliki redis image, maka perintah berikut akan secara otomatis mendownload redis image
```bash
docker run -d --name redis -p 6379:6379 redis:7.2
```
4. Jalankan eqmx, atau jika belum memiliki eqmx image, maka perintah berikut akan secara otomatis mendownload eqmx image
```bash
docker run -d --name emqx \
  -p 1883:1883 \ 
  -p 8083:8083 \   
  -p 18083:18083 \  
  emqx/emqx:latest
```

## Simulasi
Berikut merupakan langkah-langkah untuk menjalankan simulasi. Terdapat 3 terminal yang perlu dijalankan. 
- Terminal 1 adalah terminal untuk menjalankan mqtt bridge, yakni komponen yang mengelola data yang masuk dan keluar mqtt serta penulisan pada redis. cara menjalankannya dengan perintah
```bash
python -m bridge.mqtt_redis clear
```
- Terminal 2 adalah terminal untuk menjalankan searching agent, yakni komponen yang mengelola penjadwalan kendaraan listrik mulai dari pemberian jadwal, hingga pemberian akses kepada port pengisian ulang.  cara menjalankannya dengan perintah di bawah ini dengan argument genetic atau pso sesuai dengan algoritma optimasi yang ingin digunakan
```bash
python -m run_agent.run_agent [genetic | pso]
```
- Terminal 3 adalah terminal untuk menjalankan simulasi, yakni terminal untuk menjalankan berbagai skenario pengujian yang ingin dilakukan. Adapun skenario-skenario pengujian tersebut dapat melihat tabel berikut yang berisi Kode, deskripsi kasus, dan command untuk menjalankan

| **Kode**   | **Kasus**            | **Command**            |
|------------|----------------------|------------------------|
| **NFRT-01** | Tanpa Penjadwalan    | `python -m scripts.integration 0 n`  |
|            | Dengan Penjadwalan   | `python -m scripts.integration 0 y`  |
| **NFRT-02** | 40 Mobil             | `python -m scripts.integration 1 y` |
|            | 80 Mobil             | `python -m scripts.integration 2 y` |
|            | 200 Mobil            | `python -m scripts.integration 3 y`|
| **NFRT-03** | 170 KM               | `python -m scripts.integration 2 y`|
|            | 280 KM               | `python -m scripts.integration 4 y`|
|            | 430 KM               | `python -m scripts.integration 5 y`|

### Langkah - langkah simulasi

1. Buka terminal 1
2. Jalankan perintah yang digunakan pada terminal 1
3. Buka terminal 2
4. Jalankan perintah yang digunakan pada terminal 2, gunakan argument genetic untuk menggunakan algoritma optimasi genetic dan gunakan argument pso untuk menggunakan algoritma optimasi PSO
5. Buka terminal 3
6. Jalankan perintah yang digunakan pada terminal 3, gunakan perintah sesuai dengan skenario pengujian yang ingin dijalankan
7. Tunggu hingga terminal 3 selesai berjalan, atau ketika tidak ada perubahan tampilan selama lebih dari 10 menit matikan terminal 3
8. Maka akan muncul ringkasan proses simulasi
9. Visualisasi dalam bentuk hasil disimpan pada folder plots pada bagian root directory
10. Stop semua terminal, dan jalankan ulang untuk skenario berbeda

