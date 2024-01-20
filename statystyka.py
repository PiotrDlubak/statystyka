
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from datetime import date
import seaborn as sns
from scipy import stats
import math
from scipy.stats import chi2_contingency






st.set_page_config(
    page_title='Analizer Statystyka',
    layout='wide',
    page_icon=':chart_with_upwards_trend:',
    initial_sidebar_state="expanded",
    )


st.markdown("""
        <style>
               .block-container {
                    padding-top: 2rem;
                    padding-bottom: 1rem;
                    padding-left: 4rem;
                    padding-right: 4rem;
                },
            
            #MainMenu {visibility: hidden; }

        </style>
        """, unsafe_allow_html=True)



# Inicjalizacja stanu sesji dla przechowywania wybranego DataFrame
if 'df' not in st.session_state:
    st.session_state.df = None
if 'typ_ladowania' not in st.session_state:
    st.session_state.typ_ladowania = None



# Załadowanie danych demonstracyjnych
try:
    pingwin = pd.read_csv("penguins.csv")
    iris = pd.read_csv("iris.csv") 
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
    "napiwki": napiwiki

}





with st.container(border=True):

    col1, col2, col3 = st.columns([2,2,2])

    col2.header('📈 :blue[Analizer Statystyka] 	:house:', divider='rainbow')
    col2.write("")
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["O aplikacji", "Załaduj dane", "Podgąlad danych", "Ustaw parametry analizy", "Raport z analizy", "Pomoc"])
   
    with tab1:    # o aplikacji
        col1, col2= st.columns([2,2], gap = "large")
        col1.image('logo2.png')
        col2.markdown("""
            ## O Aplikacji Analizer  Statystyka ver. 1.09
            Aplikacja Statystyka jest przeznaczona do ładowania, przeglądania i analizowania zestawów danych. Umożliwia wybór różnych parametrów i miar statystycznych, co czyni ją potężnym narzędziem dla analityków danych i osób interesujących się statystyką.
            Jest ona skierowana do analityków danych, studentów, nauczycieli, a także każdego, kto interesuje się analizą danych i statystyką. Jest to szczególnie przydatne dla tych, którzy chcą zgłębić swoje umiejętności analizy danych i zrozumieć różnorodne aspekty zbiorów danych.
            - Aplikacja wykonuje analizy jednej zmiennej numerycznej, jednej zmiennej kategorialnej, dwóch zmiennych ilościowych, dwóch zmiennych kategorialnych oraz analizę zmiennej numerycznej i kategorialnej.
            - Liczy szereg miar statystycznych, w tym średnią, medianę, odchylenie standardowe, rozstęp, kwartyle, skośność, kurtozę, a także wykonuje testy statystyczne jak chi-kwadrat.
            """)
        st.divider()

        col1, col2, col3, col4 = st.columns(4)
        col1.write('Autor: Piotr Dłubak' )
        col2.write(":e-mail: statystyk@o2.pl")
        col3.write('WWW: https://piotrdlubak.github.io/')
        col4.write("GitHub :https://github.com/PiotrDlubak")

    with tab2:    # załaduj dane
        st.write("")
        typ_ladowania = st.radio("Wybierz typ ładowania danych", ['Dane demonstracyjne', 'Załaduj własny plik danych'], horizontal=True)
        st.write("")
        if typ_ladowania == 'Dane demonstracyjne':
                col1, col2 = st.columns([1,2])
                wybrane_dane = col1.selectbox("Wybierz dane demonstracyjne", list(dane_dict.keys()))
                st.session_state.df = dane_dict[wybrane_dane]
                col2.write("")        

        elif typ_ladowania == 'Załaduj własny plik danych':
                st.caption('Wymagania dot. pliku:')
                st.caption('plik może zawierać dowolną liczbę kolumn, oraz min 10 wierszy, nie może zawierać brakujących wartości')
                col1, col2 = st.columns([1,2])
                uploaded_file = col1.file_uploader("Załaduj własny plik danych (CSV, Excel)", type=['csv', 'xlsx','xls'])
                if uploaded_file is not None:
                    if uploaded_file.name.endswith('.csv'):
                        st.session_state.df = pd.read_csv(uploaded_file)
                    elif uploaded_file.name.endswith('.xlsx'):
                        st.session_state.df = pd.read_excel(uploaded_file)
                    else:
                        st.error("Nieobsługiwany format pliku. Proszę załadować plik CSV lub Excel.")

    with tab3:    # podgląd danych
        st.write("")
        if st.session_state.df is not None:
            st.dataframe(st.session_state.df)
            dframe = st.session_state.df
        else:
            st.write("Brak danych do wyświetlenia.")



    with tab4:    # parametry analizy
        kolumny_numeryczne = st.session_state.df.select_dtypes(include=[np.number]).columns
        kolumny_kategorialne = st.session_state.df.select_dtypes(exclude=[np.number]).columns
        col1, col2 = st.columns([1, 2])
        with col1:
            typ = st.selectbox("Wybierz typ analizy", ['analiza jednej zmiennej numerycznej', 'analiza jednej zmiennej kategorialnej', 
                                                    'analiza dwóch zmiennych ilościowych', 'analiza dwóch kategorialnych', 'analiza zmiennej numerycznej i kategorialnej'], key="typ_analizy")
        with col2:
            if typ == 'analiza jednej zmiennej numerycznej':
                wybrana_kolumna = st.selectbox("Wybierz kolumnę", kolumny_numeryczne)  
            
            elif typ == 'analiza jednej zmiennej kategorialnej':
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

                # Lista dostępnych typów wykresów
                typy_wykresow = ['Histogram', 'KDE', 'ECDF', 'Boxplot', 'Violinplot', 'QQ-plot']

                st.divider()

                # Dodajemy przyciski do zaznaczania i czyszczenia miar i do zaznaczania i czyszczenia wykresów

                col1, col2, col3, col4, col5 = st.columns([1,1,1,1,2])
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



                st.write(":blue[Wybierz miary]:")
                # Utworzenie checkboxów 
                for i in range(0, len(miary), 6):
                            cols = st.columns(6)
                            for j in range(6):
                                idx = i + j
                                if idx < len(miary):
                                    with cols[j]:
                                        # Sprawdzanie, czy stan checkboxa jest już zdefiniowany
                                        if miary[idx] not in st.session_state:
                                            st.session_state[miary[idx]] = False
                                        
                                        # Ustawianie checkboxa zgodnie ze stanem w st.session_state
                                        if st.checkbox(miary[idx], key=miary[idx], value=st.session_state[miary[idx]]):
                                            wybrane_miary.append(miary[idx])
                st.write(" :blue[Wybierz wykresy]:")

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

                with tab5:    # raport
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


        
        if typ == 'analiza jednej zmiennej kategorialnej':
            st.divider()
            st.write(":blue[Wybierz rodzaj raportu]:")
            tabela = st.checkbox("Tabela") 
            wykres = st.checkbox("Wykres")   

            with tab5:    # raport            
                col1, col2, col3 = st.columns([1,5,1])
                col2.subheader(f"Raport z analizy zmiennej: {wybrana_kolumna}")
                col1, col2, col3, col4 = st.columns([1,3,3,1], gap="medium")

                if tabela:
                    liczby = st.session_state.df[wybrana_kolumna].value_counts()
                    df_tabela = pd.DataFrame(liczby)
                    df_tabela.columns = ['liczba']
                    df_tabela['częstość %'] = (df_tabela['liczba'] / df_tabela['liczba'].sum()) * 100
                    col2.subheader(':blue[Tabela liczebności i częstości:]')
                    col2.dataframe(df_tabela)

                if wykres:
                    col3.subheader(':blue[Wykres liczebności i częstości:]')
                    col3.bar_chart(st.session_state.df[wybrana_kolumna].value_counts())





        if typ == 'analiza zmiennej numerycznej i kategorialnej':
            st.divider()
            st.write(" Wybierz rodzaj raportu:")
            tabela = st.checkbox("Tabela")  
            histogram = st.checkbox("Histogram")   
            kde = st.checkbox("Kde")
            box = st.checkbox("Boxplot")

          
            with tab5:
                if tabela:
                    grup = st.session_state.df.groupby(wybrana_kolumna_kategorialna)[wybrana_kolumna_numeryczna].describe().round(2)
                    st.subheader(':blue[Tabela miar zmiennej liczbowej wg poszczególych poziomów zmiennej kategorycznej:]')
                    st.write(grup)
                col1, col2, col3 = st.columns(3)
                if histogram:
                        with col1:
                            st.subheader(' Histogram')
                            fig, ax = plt.subplots(figsize=(7, 4))
                            sns.histplot(st.session_state.df, x=wybrana_kolumna_numeryczna, hue=wybrana_kolumna_kategorialna, bins=10, ax=ax)
                            st.pyplot(fig)
                if kde:
                        with col2:
                            st.subheader('KDE')
                            fig, ax = plt.subplots(figsize=(7, 4))
                            sns.kdeplot(data=st.session_state.df, x=wybrana_kolumna_numeryczna, hue=wybrana_kolumna_kategorialna, ax=ax)
                            st.pyplot(fig)

                if box:
                        with col3:
                            st.subheader(' Boxplot')
                            fig, ax = plt.subplots(figsize=(7, 4))
                            sns.boxplot(data=st.session_state.df, x=wybrana_kolumna_kategorialna, y=wybrana_kolumna_numeryczna, ax=ax)
                            st.pyplot(fig)





        if typ ==  'analiza dwóch zmiennych ilościowych':

            st.subheader(f"Raport z analizy zmiennych: {wybrana_kolumna1}  -  {wybrana_kolumna2}")
            st.divider()
            st.write("Wybierz rodzaj raportu:")
            tabela = st.checkbox("Macierz Korelacji")  
            wykres_r= st.checkbox("Wykres rozrzutu")   
            wykres_k= st.checkbox("Wykres korelacji") 
            with tab5:
                if tabela:
                    st.subheader(':blue[Macierz Korelacji:]')
                    if wybrana_kolumna1 == wybrana_kolumna2:
                        st.write("Proszę wybrać dwie różne kolumny.")
                    else:
                        corr_matrix = st.session_state.df[[wybrana_kolumna1,wybrana_kolumna2]].corr().round(2)
                        st.write(corr_matrix)
                 
                col1, col2 = st.columns(2)
                if wykres_r:
                    with col1:
                        st.subheader(':blue[Wykres rozrzutu:]')

                        def plot_scatter(df, num1, num2):
                            fig, ax = plt.subplots(figsize=(7, 4))
                            sns.regplot(data=df, x=num1, y=num2)
                            ax.set_title(f"Wykres rozrzutu: {num1} vs {num2}", color='red', fontsize=10)
                            ax.grid(axis='y', linestyle='--', color='lightgray')
                            ax.spines['top'].set_visible(False)
                            ax.spines['right'].set_visible(False)
                            ax.xaxis.label.set_visible(False)
                            ax.yaxis.label.set_size(8)
                            return fig

                        wykresy = plot_scatter(st.session_state.df, wybrana_kolumna1, wybrana_kolumna2)
                        st.pyplot(wykresy)

                if wykres_k:
                    with col2:
                        st.subheader(':blue[Wykres korelacji:]')
                        def plot_correlation_matrix(df, num1, num2):
                            fig, ax = plt.subplots(figsize=(7, 4))
                            corr = df[[num1, num2]].corr()
                            sns.heatmap(corr, annot=True, vmin=-1, vmax=1, center=0, cmap='coolwarm', 
                                        linewidths=1, linecolor='black', ax=ax)
                            ax.set_title(f"Macierz korelacji: {num1} vs {num2}", color='red', fontsize=10)
                            ax.tick_params(axis='both', which='major', labelsize=8)
                            return fig

                        wykresy = plot_correlation_matrix(st.session_state.df, wybrana_kolumna1, wybrana_kolumna2)
                        st.pyplot(wykresy)




        if typ ==  'analiza dwóch kategorialnych': 

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

            # Funkcja do tworzenia wykresów słupkowych
            def create_bar_chart(data, title):
                st.subheader(':blue[ Wykres Tabeli kontyngencji:]')
                st.bar_chart(data)

    


            st.write("")
            st.write("")
             # Tworzenie interfejsu użytkownika z checkboxami
            typy_miar = ['liczebności','częstości całkowite %',  'częstości wg kolumn %', 'częstosci wg wierszy %']
            wybrana_tabela_kontygencji = st.radio("Wybierz typ tabeli kontygencji:", typy_miar, horizontal=True)


            st.write("")
            st.write("Wybierz miary zależności:")

            # Lista dostępnych miar statystycznych
            miary_zal = ["Phi Yule'a", "C-Pearsona", "V-Cramera", "T-Czuprowa", "C- Cohena"]

            # Utworzenie pięciu kolumn
            cols = st.columns(8)

            # Przechowywanie wybranych miar
            wybrane_miary_zal = []

            # Utworzenie checkboxów w każdej kolumnie
            for i, miara in enumerate(miary_zal):
                with cols[i]:
                    if st.checkbox(miara, key=miara):
                        wybrane_miary_zal.append(miara)


            st.divider()
            with tab5:
                st.write()

                col1, col2, col3 = st.columns([1,5,1])
                col2.header(f"Raport z analizy zmiennych: {wybrana_kolumna_kat1}  -  {wybrana_kolumna_kat2}")
                st.write("")
                if wybrana_kolumna_kat1 == wybrana_kolumna_kat2:
                        st.write("Proszę wybrać dwie różne kolumny.")
                else: 
                        col1, col2 = st.columns(2)
                        col1.subheader(':blue[ Tabele kontyngencji:]')
                        if wybrana_tabela_kontygencji == 'liczebności':
                            col1.write(licz_all)
                        elif wybrana_tabela_kontygencji == 'częstości całkowite %':
                            col1.write(proc_all)
                        elif wybrana_tabela_kontygencji == 'częstości wg kolumn %':
                            col1.write(proc_col)
                        elif wybrana_tabela_kontygencji == 'częstosci wg wierszy %':
                            col1.write(proc_row)

                        # Obliczanie miar korelacji i filtrowanie wyników
                        wynik_korelacji = korelacje_nom(st.session_state.df, wybrana_kolumna_kat1, wybrana_kolumna_kat2)
                        wynik_df = wynik_korelacji[wynik_korelacji['Measure'].isin(wybrane_miary_zal)]

                        # Wyświetlanie wyników
                        col2.subheader('Wybrane miary statystyczne:')
                        col2.write(wynik_df.T)        

                        if wybrana_tabela_kontygencji == 'liczebności':
                                    create_bar_chart(licz_all_w, 'Liczebności')
                        elif wybrana_tabela_kontygencji == 'częstości całkowite %':
                                    create_bar_chart(proc_all_w, 'Częstości całkowite %')
                        elif wybrana_tabela_kontygencji == 'częstości wg kolumn %':
                                    create_bar_chart(proc_col_w, 'Częstości wg kolumn %')
                        elif wybrana_tabela_kontygencji == 'częstosci wg wierszy %':
                                    create_bar_chart(proc_row_w, 'Częstości wg wierszy %')






    with tab6:    # pomoc
        st.write("")
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

        ## Raport z analizy
        - ta sekcja pokazuje wyniki analizy dla różnych typów danych, w tym numerycznych i kategorialnych.Wybrane tabele, miary i wykresy.
 

        ## Pomoc
        Jeśli napotkasz problemy lub masz pytania dotyczące aplikacji, skorzystaj z sekcji 'Pomoc', gdzie znajdziesz odpowiedzi na często zadawane pytania oraz dodatkowe wskazówki.

        ## O aplikacji
        W tej sekcji znajdziesz informacje o aplikacji, jej wersji, autorze i celu jej stworzenia.


                        """)