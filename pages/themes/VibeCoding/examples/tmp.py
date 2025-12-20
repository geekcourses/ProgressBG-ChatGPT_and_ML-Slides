import matplotlib.pyplot as plt
import numpy as np
import calendar

# The LLM handles the specific style configurations
plt.style.use('dark_background')
months = list(calendar.month_name)[1:]
sales = np.random.randint(1000, 5000, 12)

plt.figure(figsize=(10, 6))
plt.plot(months, sales, color='#00ff41', linewidth=3, marker='o', 
         markerfacecolor='#fff', markersize=8)
plt.title("CYBER SALES 2077", fontsize=20, color='#00ff41')
plt.grid(color='#2A2A2A', linestyle='--', linewidth=0.5)
plt.show()