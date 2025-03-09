import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def create_main_df(df, start_date, end_date):
    main_df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
    return main_df

def create_station_df(df, station, start_date, end_date):
    main_df = df[(df["station"] == station) & (df["date"] >= start_date) & (df["date"] <= end_date)]
    return main_df

def create_daily_trends_df(df):
    parameters = ['PM10', 'PM2.5', 'SO2', 'CO', 'O3', 'NO2', 'pollution_index']
    daily_trends_df = df.groupby('date')[parameters].mean().reset_index()
    return daily_trends_df

def create_hourly_pattern_df(df):
    parameters = ['PM10', 'PM2.5', 'SO2', 'CO', 'O3', 'NO2', 'pollution_index']
    hourly_pattern_df = df.groupby('hour')[parameters].mean().reset_index()
    return hourly_pattern_df

def create_decomposition_df(df, pollutant):
    # Pastikan date sebagai index
    df.set_index('date', inplace=True)
    # Ambil hanya kolom pollution_index dan resample ke harian
    pollution_series = df[pollutant].resample('D').mean()
    # Lakukan decomposition
    decomposition_df = seasonal_decompose(pollution_series, model='additive', period=365)
    return decomposition_df

def create_clustered_station_pollution_df(df):
    scaler = StandardScaler()
    clustered_station_pollution_df = df.groupby('station')['pollution_index'].mean().reset_index()
    clustered_station_pollution_df["pollution_scaled"] = scaler.fit_transform(clustered_station_pollution_df[["pollution_index"]])
    # Misalnya kita pilih 3 klaster berdasarkan elbow
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clustered_station_pollution_df["cluster"] = kmeans.fit_predict(clustered_station_pollution_df[["pollution_scaled"]])
    # Buat mapping untuk mengganti angka klaster dengan label yang lebih bermakna
    cluster_labels = {
        0: "Polusi Tinggi",
        1: "Polusi Sedang",
        2: "Polusi Rendah"
    }
    # Ganti label klaster
    clustered_station_pollution_df["cluster_labels"] = clustered_station_pollution_df["cluster"].replace(cluster_labels)
    # Urutkan berdasarkan index polusi
    clustered_station_pollution_df = clustered_station_pollution_df.sort_values(by="pollution_index", ascending=False)
    return clustered_station_pollution_df

all_df = pd.read_csv("dashboard/all_data.csv.gz", compression="gzip")
all_df.sort_values(by=["station", "date","hour"], inplace=True)
all_df.reset_index(inplace=True)

all_df["date"] = pd.to_datetime(all_df["date"])

min_date = all_df["date"].min()
max_date = all_df["date"].max()
selected_station = 'Aotizhongxin'

with st.sidebar:
    st.image("dashboard/air_pollution_img.jpg")

    start_date = st.date_input("Pilih Tanggal Awal", min_value=min_date, max_value=max_date, value=min_date)
    end_date = st.date_input("Pilih Tanggal Akhir", min_value=min_date, max_value=max_date, value=max_date)

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

#======== Membuat tabel dekomposisi =========
# main_decomposition_df = create_decomposition_df(main_df, "pollution_index")
# station_decomposition_df = create_decomposition_df(station_df, "pollution_index")

st.title("Dashboard Kualitas Udara ✨⛅")

tab1, tab2 = st.tabs(["Data Keseluruhan", "Data Per Stasiun"] )

with tab1:
    st.title("Data Keseluruhan")

    #======== Membuat tabel data keseluruhan dan pola harian =========
    main_df = create_main_df(all_df, start_date, end_date)
    ovr_daily_trends_df = create_daily_trends_df(main_df)
    ovr_hourly_pattern_df = create_hourly_pattern_df(main_df)
    clustered_station_pollution_df = create_clustered_station_pollution_df(main_df)

    #==============Mengelompokkan Stasiun Berdasarkan Tingkat Polusi================
    st.header("Stasiun Berdasarkan Tingkat Polusi")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x="pollution_index", y="station", data=clustered_station_pollution_df, hue='cluster_labels', palette="Set1")
    ax.set_xlabel("Rata-rata Pollution Index")
    ax.set_ylabel("Stasiun")
    ax.set_title("Perbandingan Kualitas Udara Antar Stasiun (Berdasarkan Pollution Index)")
    ax.grid(axis="x", linestyle="--", alpha=0.7)
    ax.legend()
    
    st.pyplot(fig)

     #============= Tren harian polutan =============
    st.header("Tren Harian Stasiun Pemantauan")
    st.subheader("Tren Harian Polutan Udara")
    pollutants = ['PM10', 'PM2.5', 'SO2', 'CO', 'O3', 'NO2']

    fig, ax = plt.subplots(2, 1, figsize=(22, 16), sharex=True)
    for pol in pollutants:
        if pol != 'CO':
            ax[0].plot(ovr_daily_trends_df["date"], ovr_daily_trends_df[pol], label=pol, marker='o')
    ax[0].set_xlabel('Waktu (Hari)', fontsize=18)
    ax[0].set_ylabel('Konsentrasi Polutan (µg/m³)', fontsize=18)
    ax[0].set_title('Tren Polutan (Tanpa CO) - '+ str(selected_station), fontsize=22)
    ax[0].legend(fontsize=16)
    ax[0].grid()
    ax[0].tick_params(axis='x', rotation=45)

    ax[1].plot(ovr_daily_trends_df["date"], ovr_daily_trends_df["CO"], label="CO" ,color="red", marker='o')
    ax[1].set_xlabel('Waktu (Hari)', fontsize=18)
    ax[1].set_ylabel('Konsentrasi Polutan (µg/m³)', fontsize=18)
    ax[1].set_title('Tren Polutan CO - '+ str(selected_station), fontsize=22)
    ax[1].legend(fontsize=16)
    ax[1].grid()
    ax[1].tick_params(axis='x', rotation=45)

    st.pyplot(fig)

    st.subheader("Tren Harian Index Polusi Udara")
    fig, ax = plt.subplots(figsize=(22, 8))
    ax.plot(ovr_daily_trends_df["date"], ovr_daily_trends_df["pollution_index"], label='Pollution Index', marker='o')
    ax.set_xlabel('Waktu (Hari)', fontsize=18)
    ax.set_ylabel('Index Polusi', fontsize=18)
    ax.set_title('Tren Index Polusi - '+ str(selected_station), fontsize=22)
    ax.legend(fontsize=16)
    ax.grid()
    ax.tick_params(axis='x', rotation=45)

    st.pyplot(fig)

    #============= Pola Polutan setiap Jam =============
    st.header("Pola Polusi Setiap Jam")
    st.subheader("Pola Polutan Udara Setiap Jam")
    pollutants = ['PM10', 'PM2.5', 'SO2', 'CO', 'O3', 'NO2']

    fig, ax = plt.subplots(2, 1, figsize=(22, 16), sharex=True)
    for pol in pollutants:
        if pol != 'CO':
            ax[0].plot(ovr_hourly_pattern_df["hour"], ovr_hourly_pattern_df[pol], label=pol, marker='o')
    ax[0].set_xlabel('Waktu (Hari)', fontsize=18)
    ax[0].set_ylabel('Konsentrasi Polutan (µg/m³)', fontsize=18)
    ax[0].set_title('Pola Polutan (Tanpa CO)', fontsize=22)
    ax[0].legend(fontsize=16)
    ax[0].grid()
    ax[0].tick_params(axis='x', rotation=45)

    ax[1].plot(ovr_hourly_pattern_df["hour"], ovr_hourly_pattern_df["CO"], label="CO" ,color="red", marker='o')
    ax[1].set_xlabel('Waktu (Hari)', fontsize=18)
    ax[1].set_ylabel('Konsentrasi Polutan (µg/m³)', fontsize=18)
    ax[1].set_title('Pola Polutan CO', fontsize=22)
    ax[1].legend(fontsize=16)
    ax[1].grid()
    ax[1].tick_params(axis='x', rotation=45)

    st.pyplot(fig)

    st.subheader("Pola Index Polusi Udara setiap Jam")
    fig, ax = plt.subplots(figsize=(22, 8))
    ax.plot(ovr_hourly_pattern_df["hour"], ovr_hourly_pattern_df["pollution_index"], label='Pollution Index', marker='o')
    ax.set_xlabel('Waktu (Hari)', fontsize=18)
    ax.set_ylabel('Index Polusi', fontsize=18)
    ax.set_title('Pola Index Polusi', fontsize=22)
    ax.legend(fontsize=16)
    ax.grid()
    ax.tick_params(axis='x', rotation=45)

    st.pyplot(fig)

    #============= Dekomposisi Polutan =============
    st.header("Dekomposisi Tingkat Polusi Udara")

    try:
        ovr_decomposition_df = create_decomposition_df(main_df, "pollution_index")

        st.subheader(" Tren Polusi Udara")
        fig,ax = plt.subplots(figsize=(22, 6))
        ax.plot(ovr_decomposition_df.trend, label="Trend", color='green')
        ax.set_xlabel('Waktu (Tahun)', fontsize=18)
        ax.set_ylabel('Trend Polusi', fontsize=18)
        ax.legend(fontsize=16)

        st.pyplot(fig)

        st.subheader("Musiman Polusi Udara")
        fig,ax = plt.subplots(figsize=(22, 6))
        ax.plot(ovr_decomposition_df.seasonal, label="Seasonal", color='blue')
        ax.set_xlabel('Waktu (Tahun)', fontsize=18)
        ax.set_ylabel('Musiman Polusi', fontsize=18)
        ax.legend(fontsize=16)

        st.pyplot(fig)

        st.subheader("Residual Polusi Udara")
        fig,ax = plt.subplots(figsize=(22, 6))
        ax.plot(ovr_decomposition_df.resid, label="Residual", color='red')
        ax.set_xlabel('Waktu (Tahun)', fontsize=18)
        ax.set_ylabel('Residual Polusi', fontsize=18)
        ax.legend(fontsize=16)

        st.pyplot(fig)

    except:
         st.error("⚠️ Rentang waktu tidak cukup untuk dekomposisi! Gunakan data dengan minimal 730 observasi.")

    #============= Hubungan Faktor Cuaca dengan Indeks Polusi =============
    st.header("Hubungan Faktor Cuaca dengan Indeks Polusi")
    st.subheader("Scatter Plot Hubungan Faktor Cuaca dengan Indeks Polusi")

    weather_factors = ['TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
    target_variable = ['pollution_index']

    # Buat figure dan axes untuk subplot (2 baris, 3 kolom)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Loop untuk membuat scatter plot tiap faktor cuaca
    for i, factor in enumerate(weather_factors):
        row, col = divmod(i, 3)  # Tentukan posisi subplot
        sns.scatterplot(x=main_df[factor].squeeze(), y=main_df[target_variable].squeeze(), ax=axes[row, col])
        axes[row, col].set_title(f"{factor} vs Pollution Index")
        axes[row, col].set_xlabel(factor)
        axes[row, col].set_ylabel("Pollution Index")

    # Hapus subplot kosong jika jumlah faktor cuaca kurang dari jumlah subplot
    if len(weather_factors) < 6:
        fig.delaxes(axes[1, 2])

    fig.tight_layout(rect=[0, 0, 1, 0.96])  # Atur tata letak agar tidak saling tumpang tindih
    
    st.pyplot(fig)

    st.subheader("Korelasi antara Faktor Cuaca dengan Indeks Polusi")
    selected_columns = weather_factors + target_variable
    # Filter dataset hanya dengan kolom yang dibutuhkan
    weather_pollution_df = main_df[selected_columns].copy()
    # Hitung korelasi antar faktor cuaca dan indeks polusi
    correlation_matrix = weather_pollution_df.corr()

    # Visualisasi korelasi menggunakan heatmap
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    
    st.pyplot(fig)

with tab2:
    #======== Membuat tabel data harian dan pola harian =========
    station_df = create_station_df(all_df, selected_station, start_date, end_date)
    station_daily_trends_df = create_daily_trends_df(station_df)
    station_hourly_pattern_df = create_hourly_pattern_df(station_df)

    st.title("Data Per Stasiun")
    selected_station = st.selectbox("Pilih Stasiun", all_df["station"].unique())
    
    #============= Tren harian polutan =============
    st.header("Tren Harian Stasiun Pemantauan")
    st.subheader("Tren Harian Polutan Udara")
    pollutants = ['PM10', 'PM2.5', 'SO2', 'CO', 'O3', 'NO2']

    fig, ax = plt.subplots(2, 1, figsize=(22, 16), sharex=True)
    for pol in pollutants:
        if pol != 'CO':
            ax[0].plot(station_daily_trends_df["date"], station_daily_trends_df[pol], label=pol, marker='o')
    ax[0].set_xlabel('Waktu (Hari)', fontsize=18)
    ax[0].set_ylabel('Konsentrasi Polutan (µg/m³)', fontsize=18)
    ax[0].set_title('Tren Polutan (Tanpa CO) - '+ str(selected_station), fontsize=22)
    ax[0].legend(fontsize=16)
    ax[0].grid()
    ax[0].tick_params(axis='x', rotation=45)

    ax[1].plot(station_daily_trends_df["date"], station_daily_trends_df["CO"], label="CO" ,color="red", marker='o')
    ax[1].set_xlabel('Waktu (Hari)', fontsize=18)
    ax[1].set_ylabel('Konsentrasi Polutan (µg/m³)', fontsize=18)
    ax[1].set_title('Tren Polutan CO - '+ str(selected_station), fontsize=22)
    ax[1].legend(fontsize=16)
    ax[1].grid()
    ax[1].tick_params(axis='x', rotation=45)

    st.pyplot(fig)

    st.subheader("Tren Harian Index Polusi Udara")
    fig, ax = plt.subplots(figsize=(22, 8))
    ax.plot(station_daily_trends_df["date"], station_daily_trends_df["pollution_index"], label='Pollution Index', marker='o')
    ax.set_xlabel('Waktu (Hari)', fontsize=18)
    ax.set_ylabel('Index Polusi', fontsize=18)
    ax.set_title('Tren Index Polusi - '+ str(selected_station), fontsize=22)
    ax.legend(fontsize=16)
    ax.grid()
    ax.tick_params(axis='x', rotation=45)

    st.pyplot(fig)

    #============= Pola Polutan setiap Jam =============
    st.header("Pola Polusi Setiap Jam")
    st.subheader("Pola Polutan Udara Setiap Jam")
    pollutants = ['PM10', 'PM2.5', 'SO2', 'CO', 'O3', 'NO2']

    fig, ax = plt.subplots(2, 1, figsize=(22, 16), sharex=True)
    for pol in pollutants:
        if pol != 'CO':
            ax[0].plot(station_hourly_pattern_df["hour"], station_hourly_pattern_df[pol], label=pol, marker='o')
    ax[0].set_xlabel('Waktu (Hari)', fontsize=18)
    ax[0].set_ylabel('Konsentrasi Polutan (µg/m³)', fontsize=18)
    ax[0].set_title('Pola Polutan (Tanpa CO) - '+ str(selected_station), fontsize=22)
    ax[0].legend(fontsize=16)
    ax[0].grid()
    ax[0].tick_params(axis='x', rotation=45)

    ax[1].plot(station_hourly_pattern_df["hour"], station_hourly_pattern_df["CO"], label="CO" ,color="red", marker='o')
    ax[1].set_xlabel('Waktu (Hari)', fontsize=18)
    ax[1].set_ylabel('Konsentrasi Polutan (µg/m³)', fontsize=18)
    ax[1].set_title('Pola Polutan CO - '+ str(selected_station), fontsize=22)
    ax[1].legend(fontsize=16)
    ax[1].grid()
    ax[1].tick_params(axis='x', rotation=45)

    st.pyplot(fig)

    st.subheader("Pola Index Polusi Udara setiap Jam")
    fig, ax = plt.subplots(figsize=(22, 8))
    ax.plot(station_hourly_pattern_df["hour"], station_hourly_pattern_df["pollution_index"], label='Pollution Index', marker='o')
    ax.set_xlabel('Waktu (Hari)', fontsize=18)
    ax.set_ylabel('Index Polusi', fontsize=18)
    ax.set_title('Pola Index Polusi - '+ str(selected_station), fontsize=22)
    ax.legend(fontsize=16)
    ax.grid()
    ax.tick_params(axis='x', rotation=45)

    st.pyplot(fig)

    #============= Dekomposisi Polutan =============
    st.header("Dekomposisi Tingkat Polusi Udara")

    try:
        station_decomposition_df = create_decomposition_df(station_df, "pollution_index")

        st.subheader(" Tren Polusi Udara")
        fig,ax = plt.subplots(figsize=(22, 6))
        ax.plot(station_decomposition_df.trend, label="Trend", color='green')
        ax.set_xlabel('Waktu (Tahun)', fontsize=18)
        ax.set_ylabel('Trend Polusi', fontsize=18)
        ax.legend(fontsize=16)

        st.pyplot(fig)

        st.subheader("Musiman Polusi Udara")
        fig,ax = plt.subplots(figsize=(22, 6))
        ax.plot(station_decomposition_df.seasonal, label="Seasonal", color='blue')
        ax.set_xlabel('Waktu (Tahun)', fontsize=18)
        ax.set_ylabel('Musiman Polusi', fontsize=18)
        ax.legend(fontsize=16)

        st.pyplot(fig)

        st.subheader("Residual Polusi Udara")
        fig,ax = plt.subplots(figsize=(22, 6))
        ax.plot(station_decomposition_df.resid, label="Residual", color='red')
        ax.set_xlabel('Waktu (Tahun)', fontsize=18)
        ax.set_ylabel('Residual Polusi', fontsize=18)
        ax.legend(fontsize=16)

        st.pyplot(fig)

    except:
         st.error("⚠️ Rentang waktu tidak cukup untuk dekomposisi! Gunakan data dengan minimal 730 observasi.")



