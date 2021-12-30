import tkinter as tk

root = tk.Tk()

# menubar = tk.Menu(root)
# pageMenu = tk.Menu(menubar)
# pageMenu.add_command(label="PageOne")
# menubar.add_cascade(label="PageOne", menu=pageMenu)

root.title('Pencarian Pasal KUHP')
canvas1 = tk.Canvas(root, width=750, height=250, relief='raised')
canvas1.pack()

label1 = tk.Label(root, text='Pencarian Pasal KUHP')
label1.config(font=('helvetica', 24))
canvas1.create_window(350, 25, window=label1)

# label2 = tk.Label(root, text='Type your Number:')
# label2.config(font=('helvetica', 10))
# canvas1.create_window(200, 100, window=label2)

entry1 = tk.Entry(root)
# canvas1.create_window(200, 140, window=entry1)
entry1.place(x=125, y=70, width=550)

S = tk.Scrollbar(root)
T = tk.Text(root, height=25, width=100)
S.pack(side=tk.RIGHT, fill=tk.Y)
T.pack(side=tk.LEFT, fill=tk.Y)
S.config(command=T.yview)
T.config(yscrollcommand=S.set)

def getSquareRoot():
    x1 = entry1.get()

    label3 = tk.Label(root, text='Kasus: ' , font=('helvetica', 10))
    canvas1.create_window(350, 160, window=label3)

    label4 = tk.Label(root, text=x1, font=('helvetica', 10, 'bold'))
    canvas1.create_window(350, 180, window=label4)


    quote = """HAMLET: To be, or not to be--that is the question:
    Whether 'tis nobler in the mind to suffer
    The slings and arrows of outrageous fortune
    Or to take arms against a sea of troubles
    And by opposing end them. To die, to sleep--
    No more--and by a sleep to say we end
    The heartache, and the thousand natural shocks
    That flesh is heir to. 'Tis a consummation
    Devoutly to be wished."""
    T.insert(tk.END, quote)


button1 = tk.Button(text='Cari', command=getSquareRoot,
                    font=('helvetica', 9, 'bold'))
canvas1.create_window(350, 120, window=button1)

root.mainloop()


#