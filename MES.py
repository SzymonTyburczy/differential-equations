# problem numer 1 - rownanie transportu ciepla
import numpy as np

MIN_INTEGRATION_POINTS = 2000
LOWER_BOUND = 0.0
UPPER_BOUND = 2.0

n = int(input("Podaj n - ilosc elementow w MES: "))


def k(x):
    if 1.0 >= x >= 0.0:
        return 1
    elif 2.0 >= x > 1.0:
        return 2.0


def ei(xi, h, x):
    if xi - h <= x < xi:
        return (x - (xi - h)) / h
    elif xi <= x <= xi + h:
        return ((xi + h) - x) / h
    else:
        return 0.0


def eprim(xi, h, x):
    if xi > x >= xi - h:
        return 1.0 / h
    elif xi + h >= x >= xi:
        return -1.0 / h
    else:
        return 0.0


def main(n):
    # Alokacja tablic: macierz A, wektor wyrazów wolnych f oraz wektor rozwiązań u
    A = np.zeros((n, n), dtype=float)  # macierz kwadratowa n na n
    f_col = np.zeros(n, dtype=float)  # macierz 1 na n funkcji
    u_col = np.zeros(n, dtype=float)  # macierz 1 na n u

    # Inicjalizacja wektora prawej strony za pomoca odpowiednich wartosci
    init_result_column(f_col, n)

    # liczmy h ktore jest rowne 2/n
    h = (UPPER_BOUND - LOWER_BOUND) / n
    build_matrix(A, n, h)

    # Eliminacja Gaussa (postać schodkowa)
    echelon_form(A, f_col)

    # Rozwiązanie układu
    solve(A, f_col, u_col)

    # Zapis wyników do pliku CSV
    with open("results.csv", "w") as outfile:
        outfile.write("i,ui\n")
        for i in range(n):
            outfile.write(f"{i}, {u_col[i]}\n")

    print("Zapisano wyniki w pliku results.csv")


def init_result_column(f_col, n):
    """
    Uzupełnia kolumnę prawej strony (wyrazy wolne).
    Pierwszy węzeł -20, pozostałe 0.
    """
    f_col[0] = -20.0
    for i in range(1, n):
        f_col[i] = 0.0


def build_matrix(A, n, h):
    """
    Wypełnienie macierzy A odpowiednimi współczynnikami:
    - A[0,0] i A[0,1], A[n-1,n-2], A[n-1,n-1],
    - Wewnątrz pętli trojdiagonalnej: i-1, i, i+1.
    """
    A[0, 0] = coefficient(0, 0, h, n)
    if n > 1:
        A[0, 1] = coefficient(0, 1, h, n)

    for i in range(1, n - 1):
        for j in range(i - 1, i + 2):
            A[i, j] = coefficient(i, j, h, n)

    if n > 1:
        A[n - 1, n - 2] = coefficient(n - 1, n - 2, h, n)
        A[n - 1, n - 1] = coefficient(n - 1, n - 1, h, n)


def coefficient(row, col, h, n):
    return -ei(col * h, h, 0.0) * ei(row * h, h, 0.0) + quad_trapezoid(k, eprim, row, col, h, n)


def quad_trapezoid(f1, f3, row, col, h, n):
    """
    Całkowanie iloczynu k(x)*e'(x_i)*e'(x_j) (dla węzłów row, col)
    metodą TRAPEZÓW na TEJ SAMEJ siatce, co w oryginalnym quad_simpson.

    W oryginalnym quad_simpson mamy:
      ih = (b - a) / (integration_points + 2)
      - zaczynamy liczyć od x = a + ih
      - kończymy na x = a + (integration_points+2)*ih = b

    Tutaj robimy to samo, ale liczymy całkę metodą trapezów.
    """

    # 1. Ustalamy przedział całkowania na podstawie row, col
    lower = max(row, col) - 1 if max(row, col) != 0 else 0
    upper = min(row, col) + 1

    a = lower * h
    b = upper * h

    # 2. Liczba podprzedziałów w całkowaniu
    integration_points = max(MIN_INTEGRATION_POINTS, n)

    # 3. Krok siatki do całkowania
    ih = (b - a) / (integration_points + 2)

    # 4. Definiujemy funkcję, którą całkujemy:
    #    f(x) = k(x) * e'(x_{col},h,x) * e'(x_{row},h,x)
    def integrand(x):
        return f1(x) * f3(col * h, h, x) * f3(row * h, h, x)

    # 5. Zaczynamy od sumy wartości w punktach: x_1 = a+ih, ..., x_{m+1} = b
    #    Gdzie m = integration_points + 1
    #    Metoda trapezów:  ∫ f(x)dx ≈ (dx/2)[f(x1)+2*∑f(x2..x_m)+f(x_{m+1})]

    # Pierwszy punkt x1 i ostatni punkt x_{m+1}
    x_first = a + 1 * ih
    x_last = a + (integration_points + 1) * ih
    # (Uwaga:  x_{integration_points+2} = b)

    sum_val = integrand(x_first) + integrand(x_last)

    # Dodajemy 2 * f(xi) dla i=2..(integration_points)
    for i in range(2, integration_points + 1):
        x_i = a + i * ih
        sum_val += 2.0 * integrand(x_i)

    # Klasyczny wzór trapezów:  (dx/2) * [f(x1) + 2*... + f(x_{m+1})]
    return 0.5 * ih * sum_val


def echelon_form(A, f_col):
    """
    Przekształca macierz do postaci schodkowej (eliminacja Gaussa w wersji 'prostej').
    """
    n = A.shape[0]
    for i in range(n - 1):
        pivot = A[i, i]
        j = i + 1
        # obliczamy współczynnik proporcji do wyzerowania A[j,i]
        prop = A[j, i] / pivot if pivot != 0 else 0.0
        A[j, i] = 0.0
        # zerujemy resztę w wierszu j
        for k in range(j, n):
            A[j, k] -= prop * A[i, k]
        f_col[j] -= prop * f_col[i]


def solve(A, f_col, u_col):
    """
    Rozwiązuje układ równań A u = f (macierz już w postaci schodkowej).
    """
    n = A.shape[0]
    for i in reversed(range(n)):
        ssum = f_col[i]
        # Odejmujemy znane już składniki z prawej strony
        for j in range(i + 1, n):
            ssum -= A[i, j] * u_col[j]
        # Wyliczamy u[i]
        u_col[i] = ssum / A[i, i]


main(n)
