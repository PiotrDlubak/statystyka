
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from datetime import date
import seaborn as sns
from scipy import stats
import math
from scipy.stats import chi2_contingency




# Inicjalizacja stanu sesji dla przechowywania wybranego DataFrame
if 'df' not in st.session_state:
    st.session_state.df = None
if 'typ_ladowania' not in st.session_state:
    st.session_state.typ_ladowania = None



# Załadowanie danych demonstracyjnych
try:
    pingwin = pd.read_csv("penguins.csv")
    iris = pd.read_csv("iris.csv") 
    auta = pd.read_excel("auta.xlsx")
    napiwiki = pd.read_csv("tips.csv")
    szkoła = pd.read_excel("szkoła.xlsx")
    sklep = pd.read_excel("sklepy.xlsx")

except FileNotFoundError:
    st.error("Nie znaleziono plików danych. Upewnij się, że pliki są w katalogu roboczym.")

dane_dict = {

    "sklepy": sklep,
    "szkoła" :szkoła,
    "pingwin": pingwin,
    "iris": iris,
    "samochody": auta,
    "napiwki": napiwiki

}



st.set_page_config(layout="wide")

st.markdown("""
        <style>
               .block-container {
                    padding-top: 2rem;
                    padding-bottom: 2rem;
                    padding-left: 3rem;
                    padding-right: 3rem;
                }
        </style>
        """, unsafe_allow_html=True)



if 'is_first_run' not in st.session_state:
    st.session_state['is_first_run'] = True

# Wykonaj te instrukcje tylko, gdy aplikacja jest uruchamiana po raz pierwszy
if st.session_state['is_first_run']:

    col1, col2, col3 = st.columns([2,2,2])
    col2.header(' :blue[Analizer Statystyka] ')
    col1, col2, col3 = st.columns([1,5,1])
    col2.image('logo2.png', width=800)

    # Ustaw is_first_run na False, aby nie wykonywać tego ponownie
    st.session_state['is_first_run'] = False
    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    col1. write('Autor: Piotr Dłubak' )
    col2.write(":e-mail: statystyk@o2.pl")
    col3.write('WWW: https://piotrdlubak.github.io/')
    col4.write("GitHub :https://github.com/PiotrDlubak")





# Tworzenie paska bocznego z nagłówkami i przyciskami
sidebar = st.sidebar

# Inicjalizacja stanu sesji dla przechowywania wybranego DataFrame
if 'df' not in st.session_state:
    st.session_state.df = None
if 'typ_ladowania' not in st.session_state:
    st.session_state.typ_ladowania = None


if not st.session_state:
    st.title("start")

sidebar.header(':blue[Analizer Statystyka    ver. 1.09:chart_with_upwards_trend:]' , divider='rainbow')


if sidebar.button('Źródło danych', key='zrodlo'):
    st.session_state.typ_ladowania = 'zrodlo'

if sidebar.button('Podgląd danych', key='podglad'):
    st.session_state.typ_ladowania = 'podglad'

if sidebar.button('Parametry analizy', key='parametry'):
    st.session_state.typ_ladowania = 'parametry'

if sidebar.button('Pomoc statystyka', key='pomoc'):
    st.session_state.typ_ladowania = 'pomoc'

if sidebar.button('O aplikacji', key='aplikacja'):
    st.session_state.typ_ladowania = 'aplikacja'


# Wyświetlanie odpowiedniej sekcji w zależności od wybranego typu ładowania
    
if st.session_state.typ_ladowania == 'zrodlo':
    typ_ladowania = st.radio("Wybierz typ ładowania danych", ['Dane demonstracyjne', 'Załaduj własny plik danych'], horizontal=True)
    st.divider()
    if typ_ladowania == 'Dane demonstracyjne':
        wybrane_dane = st.selectbox("Wybierz dane demonstracyjne", list(dane_dict.keys()))
        st.session_state.df = dane_dict[wybrane_dane]
        st.write(f"Wybrane dane demonstracyjne: {wybrane_dane}")

                

    elif typ_ladowania == 'Załaduj własny plik danych':
        st.caption('Wymagania dot. pliku:')
        st.caption('plik może zawierać dowolną liczbę kolumn, oraz min 10 wierszy, nie może zawierać brakujących wartości')

        uploaded_file = st.file_uploader("Lub załaduj własny plik danych (CSV, Excel)", type=['csv', 'xlsx','xls'])
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.csv'):
                st.session_state.df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                st.session_state.df = pd.read_excel(uploaded_file)
            else:
                st.error("Nieobsługiwany format pliku. Proszę załadować plik CSV lub Excel.")

    #elif typ_ladowania == 'Stwórz dane wewnętrzne': pass

    
elif st.session_state.typ_ladowania == 'podglad':
    st.divider()
    # Przycisk do wyświetlania DataFrame
    if st.session_state.df is not None:
        st.dataframe(st.session_state.df)
        dframe = st.session_state.df
    else:
        st.write("Brak danych do wyświetlenia.")
        
    # if 'df' in st.session_state and st.session_state.df is not None:
    #     dframe = st.session_state.df
    #     st.dataframe(dframe)

    #     # Tworzenie słownika na filtry
    #     filtry = {}

    #     # Tworzenie filtrów dla każdej kolumny
    #     for col in dframe.columns:
    #         unikalne_wartosci = pd.unique(dframe[col])
    #         wybrane_wartosci = st.multiselect(f"Filtruj {col}", unikalne_wartosci, default=unikalne_wartosci)
    #         filtry[col] = wybrane_wartosci

    #     # Filtrowanie DataFrame
    #     for col, wartosci in filtry.items():
    #         dframe = dframe[dframe[col].isin(wartosci)]

    #     # Wyświetlenie przefiltrowanego DataFrame
    #     st.write("Przefiltrowane dane:")
    #     st.dataframe(dframe)
    # else:
    #     st.write("Brak danych do wyświetlenia.")


elif st.session_state.typ_ladowania == 'parametry':

    kolumny_numeryczne = st.session_state.df.select_dtypes(include=[np.number]).columns
    kolumny_kategorialne = st.session_state.df.select_dtypes(exclude=[np.number]).columns

    # Wyświetlanie opcji analizy
    
    col1, col2 = st.columns([1, 2])
    with col1:
        typ = st.selectbox("Wybierz typ analizy", ['analiza jednej zmiennej numerycznej', 'analia jednej zmiennej kategorialnej', 
                                                   'analiza dwóch zmiennych ilościowych', 'analiza dwóch kategorialnych', 'analiza zmiennej numerycznej i kategorialnej'], key="typ_analizy")
    with col2:
        if typ == 'analiza jednej zmiennej numerycznej':
            wybrana_kolumna = st.selectbox("Wybierz kolumnę", kolumny_numeryczne)  
        
        elif typ == 'analia jednej zmiennej kategorialnej':
            wybrana_kolumna = st.selectbox("Wybierz kolumnę", kolumny_kategorialne)

        elif typ == 'analiza dwóch zmiennych ilościowych':
            col1, col2 = st.columns(2)
            with col1:
                wybrana_kolumna1 = st.selectbox("Wybierz kolumnę1", kolumny_numeryczne, key="col1_num")
            with col2:
                wybrana_kolumna2 = st.selectbox("Wybierz kolumnę2", kolumny_numeryczne, key="col2_num")
            if wybrana_kolumna1 == wybrana_kolumna2:
                st.write("Proszę wybrać dwie różne kolumny.")

        elif typ == 'analiza dwóch kategorialnych':
            col1, col2 = st.columns(2)
            with col1:
                wybrana_kolumna_kat1 = st.selectbox("Wybierz kolumnę1", kolumny_kategorialne, key="col1_cat")
            with col2:
                wybrana_kolumna_kat2 = st.selectbox("Wybierz kolumnę2", kolumny_kategorialne, key="col2_cat")
            if wybrana_kolumna_kat1 == wybrana_kolumna_kat2:
                st.write("Proszę wybrać dwie różne kolumny.")

        elif typ == 'analiza zmiennej numerycznej i kategorialnej':
            col1, col2 = st.columns(2)
            with col1:
                wybrana_kolumna_numeryczna = st.selectbox("Wybierz kolumnę numeryczną", kolumny_numeryczne, key="col_num")
            with col2:
                wybrana_kolumna_kategorialna = st.selectbox("Wybierz kolumnę kategorialną", kolumny_kategorialne, key="col_cat")

    

    if typ == 'analiza jednej zmiennej numerycznej':

        def stat(df, wybrane_miary):
                    wyniki = {
                        'liczba': df.count(),
                        'liczba brakujących': df.isna().sum(),
                        'suma': df.sum(),
                        'min': df.min(),
                        'max': df.max(),
                        'średnia': df.mean(),
                        'rozstęp': df.max() - df.min(),
                        'Q_10%': df.quantile(.1),
                        'Q1_25%': df.quantile(.25),
                        'Q2_50%': df.quantile(.5),
                        'Q3_75%': df.quantile(.75),
                        'Q_90%': df.quantile(.9),
                        'IQR': df.quantile(.75) - df.quantile(.25),
                        'odch_cwiar': (df.quantile(.75) - df.quantile(.25)) / 2,
                        'odchylenie przeciętne': np.mean(np.abs(df - df.mean())) / df.mean() * 100,
                        'wariancja': df.var(ddof=1),
                        'odch_std': df.std(ddof=1),
                        'błąd_odch_std':df.std(ddof=1) / np.sqrt(df.count()),
                        'kl_wsp_zmien': df.std(ddof=1) / df.mean(),
                        'poz_wsp_zmien': (df.quantile(.75) - df.quantile(.25)) / df.quantile(.5),
                        'skośność': df.skew(),
                        'kurtoza': df.kurtosis()
                        }
                    wyniki_wybrane = {miara: wyniki[miara].round(2) for miara in wybrane_miary}
                    return wyniki_wybrane

# Funkcja generująca wykresy
        def generuj_wykresy_streamlit(df, kolumna, wybrane_wykresy):
                    clean_data = pd.to_numeric(df[kolumna], errors='coerce').dropna()
                    if wybrane_wykresy:
                        # Podział wykresów na dwie grupy
                        wykresy_grupa_1 = wybrane_wykresy[:3]
                        wykresy_grupa_2 = wybrane_wykresy[3:]
                    
                        # Tworzenie pierwszego zestawu trzech kolumn
                        cols_1 = st.columns(3)
                        for i, wykres in enumerate(wykresy_grupa_1):
                            with cols_1[i]:
                                fig, ax = plt.subplots(figsize=(5, 4))
                                # Logika generowania wykresów
                                if wykres == 'Histogram':
                                    sns.histplot(clean_data, bins=5,ax=ax)
                                    ax.set_title('Histogram')
                                elif wykres == 'KDE':
                                    sns.kdeplot(clean_data, ax=ax)
                                    ax.set_title('KDE')
                                elif wykres == 'ECDF':
                                    sns.ecdfplot(clean_data, ax=ax)
                                    ax.set_title('ECDF')
                                elif wykres == 'Boxplot':
                                    sns.boxplot(x=clean_data, ax=ax)
                                    ax.set_title('Boxplot')
                                elif wykres == 'Violinplot':
                                    sns.violinplot(x=clean_data, ax=ax)
                                    ax.set_title('Violinplot')
                                elif wykres == 'QQ-plot':
                                    stats.probplot(clean_data.dropna(), dist="norm", plot=ax)
                                    ax.set_title('QQ-plot')
                                st.pyplot(fig)

                        # Tworzenie drugiego zestawu trzech kolumn, jeśli jest więcej niż trzy wykresy
                        if wykresy_grupa_2:
                            st.write("")  # Dodaj pusty wiersz dla lepszego rozdzielenia grup wykresów
                            cols_2 = st.columns(3)
                            for i, wykres in enumerate(wykresy_grupa_2):
                                with cols_2[i]:
                                    fig, ax = plt.subplots(figsize=(5, 4))
                                    # Logika generowania wykresów
                                    if wykres == 'Histogram':
                                        sns.histplot(clean_data, ax=ax)
                                        ax.set_title('Histogram')
                                    elif wykres == 'KDE':
                                        sns.kdeplot(clean_data, ax=ax)
                                        ax.set_title('KDE')
                                    elif wykres == 'ECDF':
                                        sns.ecdfplot(clean_data, ax=ax)
                                        ax.set_title('ECDF')
                                    elif wykres == 'Boxplot':
                                        sns.boxplot(x=clean_data, ax=ax)
                                        ax.set_title('Boxplot')
                                    elif wykres == 'Violinplot':
                                        sns.violinplot(x=clean_data, ax=ax)
                                        ax.set_title('Violinplot')
                                    elif wykres == 'QQ-plot':
                                        stats.probplot(clean_data.dropna(), dist="norm", plot=ax)
                                        ax.set_title('QQ-plot')
                                    st.pyplot(fig)





        # Lista dostępnych miar statystycznych
        miary = ['liczba', 'liczba brakujących', 'suma', 'min', 'max', 'średnia', 'rozstęp', 'Q_10%', 'Q1_25%', 'Q2_50%', 'Q3_75%', 'Q_90%', 'IQR', 
                        'odch_cwiar', 'odchylenie przeciętne', 'wariancja', 'odch_std', 'błąd_odch_std', 'kl_wsp_zmien', 'poz_wsp_zmien', 'skośność', 'kurtoza']
        wybrane_miary = []

        st.divider()
        # Lista dostępnych typów wykresów
        typy_wykresow = ['Histogram', 'KDE', 'ECDF', 'Boxplot', 'Violinplot', 'QQ-plot']


        st.write("")
        # Dodajemy przyciski do zaznaczania i czyszczenia miar i do zaznaczania i czyszczenia wykresów
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button('Zaznacz wszystkie miary'):
                        for miara in miary:
                            st.session_state[miara] = True
        with col2:                    
            if st.button('Wyczyść wszystkie miary'):
                        for miara in miary:
                            st.session_state[miara] = False
        with col3:                    
            if st.button('Zaznacz wszystkie wykresy'):
                        for wykres in typy_wykresow:
                            st.session_state[wykres] = True
        with col4:
            if st.button('Wyczyść wszystkie wykresy'):
                        for wykres in typy_wykresow:
                            st.session_state[wykres] = False
        st.write("Wybierz miary:")

        # Utworzenie checkboxów 
        for i in range(0, len(miary), 5):
                    cols = st.columns(5)
                    for j in range(5):
                        idx = i + j
                        if idx < len(miary):
                            with cols[j]:
                                # Sprawdzanie, czy stan checkboxa jest już zdefiniowany
                                if miary[idx] not in st.session_state:
                                    st.session_state[miary[idx]] = False
                                
                                # Ustawianie checkboxa zgodnie ze stanem w st.session_state
                                if st.checkbox(miary[idx], key=miary[idx], value=st.session_state[miary[idx]]):
                                    wybrane_miary.append(miary[idx])
        st.write("Wybierz wykresy:")

        # Tworzenie checkboxów dla wyboru typów wykresów
        wybrane_wykresy = []
    
        for i in range(0, len(typy_wykresow), 6):
            # Tworzenie trzech kolumn
            cols = st.columns(6)

            for j in range(6):
                idx = i + j
                # Sprawdzenie, czy indeks nie przekracza długości listy
                if idx < len(typy_wykresow):
                    with cols[j]:
                        # Sprawdzanie, czy stan checkboxa jest już zdefiniowany
                        if typy_wykresow[idx] not in st.session_state:
                            st.session_state[typy_wykresow[idx]] = False
                        
                        # Ustawianie checkboxa zgodnie ze stanem w st.session_state
                        if st.checkbox(typy_wykresow[idx], key=typy_wykresow[idx], value=st.session_state[typy_wykresow[idx]]):
                            wybrane_wykresy.append(typy_wykresow[idx])

        st.divider()

        # Dodanie przycisku "Wykonaj analizę"
        #if st.button('Wykonaj analizę '):

        col1, col2, col3 = st.columns([1,5,1])
        col2.header(f"Raport z analizy zmiennej: {wybrana_kolumna}")
        st.subheader(':blue[Tabela rozdzielcza:]')

        tab = pd.DataFrame(st.session_state.df[wybrana_kolumna].value_counts(bins=5).sort_index().round(2))
        tab['licz skum'] = tab['count'].cumsum()
        tab['częstość'] = tab['count'] / (tab['count'].sum()) * 100
        tab['częstość skum'] = tab['częstość'].cumsum()
        st.dataframe(tab)

        wynik = stat(st.session_state.df[wybrana_kolumna], wybrane_miary)
    
        wynik_df = pd.DataFrame([wynik])
        
        wynik_part1 = wynik_df.iloc[:, :12]  # Pierwsze 12 kolumn
        wynik_part2 = wynik_df.iloc[:, 12:]  # Pozostałe kolumny

        st.subheader(':blue[Wartości wybranych miar statystycznych:]')
        st.write(wynik_part1)
        st.write(wynik_part2)

        st.subheader(':blue[Wykresy:]')
        generuj_wykresy_streamlit(st.session_state.df, wybrana_kolumna, wybrane_wykresy)

  

    if typ ==  'analia jednej zmiennej kategorialnej':
        st.divider()
        col1, col2, col3 = st.columns([1,5,1])
        col2.header(f"Raport z analizy zmiennej: {wybrana_kolumna}")
        col1,col2 = st.columns(2)
        tabela = pd.DataFrame(st.session_state.df[wybrana_kolumna].value_counts())
        tabela['częstość %']=(tabela['count']/(tabela['count'].sum())*100)
        col1.subheader(':blue[Tabela liczebności i częstości:]')
        col1.dataframe(tabela)
        col2.subheader(':blue[Wykres liczebności i częstości:]')
        plt.figure(figsize=(6, 3))
        st.session_state.df[wybrana_kolumna].value_counts().plot(kind = 'barh',linewidth=0.8, title=f'Rozkład zmiennej: [{wybrana_kolumna}]')
        col2.pyplot(plt)


    if typ ==  'analiza zmiennej numerycznej i kategorialnej':
        st.divider()
        col1, col2, col3 = st.columns([1,5,1])
        col2.header(f"Raport z analizy zmiennych: {wybrana_kolumna_numeryczna}  -  {wybrana_kolumna_kategorialna}")
        grup = st.session_state.df.groupby(wybrana_kolumna_kategorialna)[wybrana_kolumna_numeryczna].describe().round(2)
        st.subheader(':blue[Tabela miar zmiennej liczwowej wg poszczególych poziomów zmiennej kategorycznej:]')
        st.write(grup)
        st.subheader(':blue[Wybrane wykresy :]')
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))

        # Histogram
        axes[0].set_title("Histogram - liczebności", color='red', fontsize=10)
        sns.histplot(st.session_state.df, x=wybrana_kolumna_numeryczna, stat='count', hue=wybrana_kolumna_kategorialna, bins=9, fill=True, color="black", alpha=0.6, ax=axes[0])

        # KDE
        axes[1].set_title("KDE", color='red', fontsize=10)
        sns.kdeplot(st.session_state.df, x=wybrana_kolumna_numeryczna, hue=wybrana_kolumna_kategorialna, label=True, ax=axes[1])

        # Boxplot
        axes[2].set_title("Boxplot", color='red', fontsize=10)
        sns.boxplot(data=st.session_state.df, x=wybrana_kolumna_numeryczna, y=wybrana_kolumna_kategorialna, ax=axes[2], orient='h')

        # Wyświetlenie wykresów w Streamlit
        st.pyplot(fig)


    if typ ==  'analiza dwóch zmiennych ilościowych':
        col1, col2, col3 = st.columns([1,5,1])
        col2.header(f"Raport z analizy zmiennych: {wybrana_kolumna1}  -  {wybrana_kolumna2}")
        st.subheader(':blue[Macierz Korelacji:]')
        if wybrana_kolumna1 == wybrana_kolumna2:
            st.write("Proszę wybrać dwie różne kolumny.")
        else:
            corr_matrix = st.session_state.df[[wybrana_kolumna1,wybrana_kolumna2]].corr().round(2)
            st.write(corr_matrix)

        st.subheader(':blue[Wykres rozrzutu i  korelacji:]')


        def num_plot2(df, num1, num2):
            fig, axs = plt.subplots(1, 2, figsize=(14, 4))
            corr = df[[num1, num2]].corr()
            plt.suptitle(f'Wykres zależności: {num1} : {num2}', fontsize=9, color='black')
            
            axs[0].set_title("Wykres rozrzutu", color='red', fontsize=10)
            sns.regplot(data=df, x=num1, y=num2, ax=axs[0])
            
            axs[1].set_title("Wykres korelacji", color='red', fontsize=10)
            sns.heatmap(corr, annot=True, vmin=-1, vmax=1, center=0, cmap='coolwarm', 
                        linewidths=1, linecolor='black', ax=axs[1])

            for ax in axs:
                ax.tick_params(axis='both', which='major', labelsize=8)
                ax.grid(axis='y', linestyle='--', color='lightgray')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.xaxis.label.set_visible(False)
                ax.yaxis.label.set_size(8)

            plt.tight_layout()
            return fig

        wykresy = num_plot2(st.session_state.df, wybrana_kolumna1, wybrana_kolumna2)
        st.pyplot(wykresy)


    if typ ==  'analiza dwóch kategorialnych':
        st.divider()
        col1, col2, col3 = st.columns([1,5,1])
        col2.header(f"Raport z analizy zmiennych: {wybrana_kolumna_kat1}  -  {wybrana_kolumna_kat2}")

        if wybrana_kolumna_kat1 == wybrana_kolumna_kat2:
            st.write("Proszę wybrać dwie różne kolumny.")
        else: 
            col1, col2, col3 = st.columns([1,5,1])
            st.subheader(':blue[ Tabele kontyngencji:]')




        def rozklady_cat(df, cat1, cat2, widok):

            if widok == 'licz_all':
                print('liczebności:')
                t = pd.crosstab(df[cat1], df[cat2], margins=True, margins_name="Razem")
            elif widok == 'proc_all': 
                print('częstości całkowite %:')
                t = (pd.crosstab(df[cat1], df[cat2], dropna=False, normalize='all', margins=True, margins_name='suma')*100).round(2)
            elif widok == 'proc_col':
                print('częstości wg kolumn %: ')
                t = (pd.crosstab(df[cat1], df[cat2], dropna=False, normalize='columns')*100).round(2)
            elif widok == 'proc_row':
                print('częstosci wg wierszy %:')
                t = (pd.crosstab(df[cat1], df[cat2], dropna=False, normalize='index', margins_name='suma')*100).round(2)
            return t

        licz_all = rozklady_cat(st.session_state.df, wybrana_kolumna_kat1, wybrana_kolumna_kat2, 'licz_all')
        proc_all = rozklady_cat(st.session_state.df, wybrana_kolumna_kat1, wybrana_kolumna_kat2, 'proc_all')
        proc_col = rozklady_cat(st.session_state.df, wybrana_kolumna_kat1, wybrana_kolumna_kat2, 'proc_col')
        proc_row = rozklady_cat(st.session_state.df, wybrana_kolumna_kat1, wybrana_kolumna_kat2, 'proc_row')
        

        def rozklady_cat_w(df, cat1, cat2, widok):

            if widok == 'licz_all':
                print('liczebności:')
                t = pd.crosstab(df[cat1], df[cat2])
            elif widok == 'proc_all': 
                print('częstości całkowite %:')
                t = (pd.crosstab(df[cat1], df[cat2], dropna=False, normalize='all')*100).round(2)
            elif widok == 'proc_col':
                print('częstości wg kolumn %: ')
                t = (pd.crosstab(df[cat1], df[cat2], dropna=False, normalize='columns')*100).round(2)
            elif widok == 'proc_row':
                print('częstosci wg wierszy %:')
                t = (pd.crosstab(df[cat1], df[cat2], dropna=False, normalize='index')*100).round(2)
            return t

        licz_all_w = rozklady_cat_w(st.session_state.df, wybrana_kolumna_kat1, wybrana_kolumna_kat2, 'licz_all')
        proc_all_w = rozklady_cat_w(st.session_state.df, wybrana_kolumna_kat1, wybrana_kolumna_kat2, 'proc_all')
        proc_col_w = rozklady_cat_w(st.session_state.df, wybrana_kolumna_kat1, wybrana_kolumna_kat2, 'proc_col')
        proc_row_w = rozklady_cat_w(st.session_state.df, wybrana_kolumna_kat1, wybrana_kolumna_kat2, 'proc_row')
        

        
    # Tworzenie interfejsu użytkownika z checkboxami
        typy_miar = ['liczebności','częstości całkowite %',  'częstości wg kolumn %', 'częstosci wg wierszy %']
        wybrana_tabela_kontygencji = st.radio("Wybierz typ tabeli kontygencji:", typy_miar, horizontal=True)
        
        st.write()
        if wybrana_tabela_kontygencji == 'liczebności':
            st.write(licz_all)
        elif wybrana_tabela_kontygencji == 'częstości całkowite %':
            st.write(proc_all)
        elif wybrana_tabela_kontygencji == 'częstości wg kolumn %':
            st.write(proc_col)
        elif wybrana_tabela_kontygencji == 'częstosci wg wierszy %':
            st.write(proc_row)
            


        # Funkcja do tworzenia wykresów słupkowych
        def create_bar_chart(data, title):
            st.subheader(':blue[ Wykres Tabeli kontyngencji:]')
            st.bar_chart(data)

    
        if wybrana_tabela_kontygencji == 'liczebności':
            create_bar_chart(licz_all_w, 'Liczebności')
        elif wybrana_tabela_kontygencji == 'częstości całkowite %':
            create_bar_chart(proc_all_w, 'Częstości całkowite %')
        elif wybrana_tabela_kontygencji == 'częstości wg kolumn %':
            create_bar_chart(proc_col_w, 'Częstości wg kolumn %')
        elif wybrana_tabela_kontygencji == 'częstosci wg wierszy %':
            create_bar_chart(proc_row_w, 'Częstości wg wierszy %')






        def korelacje_nom(df, var1, var2):
            """ 'phi' - phi Yule'a, 'cp' - C-Pearsona, 'v' - V-Cramera,'t' - T-Czuprowa, 'c' - Cohena """
            table = pd.crosstab(df[var1], df[var2])
            df2 = df[[var1, var2]]
            chi2, p, dof, expected = chi2_contingency(table)
            N = df.shape[0]  # liczba elementow/obserwacji
            r = table.shape[0]  # liczba wierszy w tabeli kontyngencji.
            k = table.shape[1]  # liczba kolumn w tabeli kontyngencji,
            phi = (chi2 / N).round(3)
            C_pearson = np.round(math.sqrt(chi2 / (chi2 + N)), 3)
            V_cramer = np.round(math.sqrt(chi2 / (N * min(k - 1, r - 1))), 3)
            T_czuprow = np.round(math.sqrt(chi2 / (N * np.sqrt((r - 1) * (k - 1)))), 3)
            Cohen = np.round(V_cramer * (math.sqrt(min(k - 1, r - 1) - 1)), 3)
            
            # Tworzenie listy wyników
            results = [{'Measure': 'Phi Yule\'a', 'Value': phi},
                    {'Measure': 'C-Pearsona', 'Value': C_pearson},
                    {'Measure': 'V-Cramera', 'Value': V_cramer},
                    {'Measure': 'T-Czuprowa', 'Value': T_czuprow},
                    {'Measure': 'Cohena', 'Value': Cohen}]
            
            # Tworzenie ramki danych z wynikami
            result_df = pd.DataFrame(results)
            
            return result_df



        # Lista dostępnych miar statystycznych
        miary_zal = ["Phi Yule'a", "C-Pearsona", "V-Cramera", "T-Czuprowa", "C- Cohena"]

        # Utworzenie pięciu kolumn
        cols = st.columns(5)

        # Przechowywanie wybranych miar
        wybrane_miary_zal = []

        # Utworzenie checkboxów w każdej kolumnie
        for i, miara in enumerate(miary_zal):
            with cols[i]:
                if st.checkbox(miara, key=miara):
                    wybrane_miary_zal.append(miara)

        # Obliczanie miar korelacji i filtrowanie wyników
        wynik_korelacji = korelacje_nom(st.session_state.df, wybrana_kolumna_kat1, wybrana_kolumna_kat2)
        wynik_df = wynik_korelacji[wynik_korelacji['Measure'].isin(wybrane_miary_zal)]

        # Wyświetlanie wyników
        st.subheader('Wybrane miary statystyczne:')
        st.write(wynik_df.T)


if st.session_state['typ_ladowania'] == 'pomoc':


     # Wyświetlanie szczegółowych informacji o funkcjonalnościach
    st.markdown("""
    ## Jak korzystać z aplikacji

Aplikacja Statystyka jest intuicyjnym narzędziem do analizy danych. Oto krótki przewodnik, jak z niej korzystać:

## Źródło danych
W tej sekcji możesz wybrać sposób ładowania danych:
- **Dane demonstracyjne**: Wybierz z predefiniowanych zestawów danych.
- **Załaduj własny plik danych**: Możesz załadować swoje dane w formacie CSV lub Excel.

## Podgląd danych
Po załadowaniu danych, możesz zobaczyć podgląd swojego DataFrame. Pokazuje to, jak dane są strukturyzowane.

## Parametry analizy
Ta sekcja pozwala na wybór różnych parametrów i miar statystycznych do analizy wybranych danych.

## Analiza
- Możesz wykonywać analizy dla różnych typów danych, w tym numerycznych i kategorialnych.
- Aplikacja oferuje różne rodzaje analiz, takie jak analiza jednej zmiennej, analiza dwóch zmiennych, a także testy statystyczne.

## Wykresy i wizualizacje
- Aplikacja umożliwia tworzenie różnorodnych wykresów statystycznych, które pomagają wizualizować dane i wyniki analiz.
- Możesz generować histogramy, wykresy pudełkowe, wykresy rozrzutu i wiele innych.

## Pomoc
Jeśli napotkasz problemy lub masz pytania dotyczące aplikacji, skorzystaj z sekcji 'Pomoc', gdzie znajdziesz odpowiedzi na często zadawane pytania oraz dodatkowe wskazówki.

## O aplikacji
W tej sekcji znajdziesz informacje o aplikacji, jej wersji, autorze i celu jej stworzenia.


                
""")




if st.session_state['typ_ladowania'] == 'aplikacja':

    st.markdown("""
    ## O Aplikacji Analizer  Statystyka ver. 1.09
    
 

Aplikacja Statystyka jest przeznaczona do ładowania, przeglądania i analizowania zestawów danych. Umożliwia wybór różnych parametrów i miar statystycznych, co czyni ją potężnym narzędziem dla analityków danych i osób interesujących się statystyką.

## Dla kogo jest ta aplikacja

Jest ona skierowana do analityków danych, studentów, nauczycieli, a także każdego, kto interesuje się analizą danych i statystyką. Jest to szczególnie przydatne dla tych, którzy chcą zgłębić swoje umiejętności analizy danych i zrozumieć różnorodne aspekty zbiorów danych.

## Rodzaje Analiz i Miary Statystyczne

- Aplikacja wykonuje analizy jednej zmiennej numerycznej, jednej zmiennej kategorialnej, dwóch zmiennych ilościowych, dwóch zmiennych kategorialnych oraz analizę zmiennej numerycznej i kategorialnej.
- Liczy szereg miar statystycznych, w tym średnią, medianę, odchylenie standardowe, rozstęp, kwartyle, skośność, kurtozę, a także wykonuje testy statystyczne jak chi-kwadrat.

## Funkcjonalności Aplikacji

### Ładowanie Danych

Użytkownicy mogą ładować dane demonstracyjne, własne pliki CSV/Excel, a także tworzyć dane bezpośrednio w aplikacji.

### Podgląd Danych

Po wczytaniu danych można je przeglądać w aplikacji, co ułatwia zrozumienie ich struktury i zawartości.

### Analiza Statystyczna

Użytkownicy mogą wybierać różne miary statystyczne i generować wykresy, co pomaga w głębszej analizie danych.

### Interaktywny Interfejs

Aplikacja oferuje intuicyjny i łatwy w obsłudze interfejs użytkownika, co ułatwia analizę nawet dla osób niebędących ekspertami w danych.

### Pomoc i Informacje o Aplikacji

Sekcje 'Pomoc' i 'O Aplikacji' dostarczają użytkownikowi niezbędnych informacji i wsparcia.

""")
    
    st.divider()
    st. write('Autor: Piotr Dłubak')
    st. write('Kontakt:')
    st.write(" e-mail :e-mail: statystyk@o2.pl")
    st.write('www: https://piotrdlubak.github.io/')
    st.write(" GitHub :https://github.com/PiotrDlubak")

 