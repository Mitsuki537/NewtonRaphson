# -*- coding: utf-8 -*-


import tkinter as tk
from tkinter import messagebox
import subprocess

# Diccionario para mapear opción con script
scripts = {
    "2 variables": "NR-enl.py",
    "3 variables": "nr3.py",
    "4 variables": "nr4.py",
    "5 variables": "nr5.py"
}

def ejecutar_script(opcion):
    archivo = scripts.get(opcion)
    if archivo:
        try:
            subprocess.run(["python", archivo], check=True)
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Error", f"Ocurrió un error al ejecutar {archivo}:\n{e}")
    else:
        messagebox.showwarning("Advertencia", "Por favor selecciona una opción válida.")

# Crear ventana
ventana = tk.Tk()
ventana.title("Método de Newton-Raphson")
ventana.geometry("350x250")
ventana.config(bg="#e6f2ff")

# Título
titulo = tk.Label(ventana, text="Selecciona el número de variables", font=("Arial", 14), bg="#e6f2ff")
titulo.pack(pady=20)

# Menú desplegable
opciones = list(scripts.keys())
variable = tk.StringVar(ventana)
variable.set(opciones[0])

menu = tk.OptionMenu(ventana, variable, *opciones)
menu.config(font=("Arial", 12))
menu.pack(pady=10)

# Botón de ejecutar
boton = tk.Button(ventana, text="Ejecutar", font=("Arial", 12), bg="#4CAF50", fg="white",
                  command=lambda: ejecutar_script(variable.get()))
boton.pack(pady=20)

ventana.mainloop()
