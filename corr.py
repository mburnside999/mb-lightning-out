import pandas as pd


    # Sample data in a Pandas DataFrame
data = {
        'name': ['John', 'Alice', 'Bob'],
        'age': [30, 25, 28],
        'department': ['HR', 'Finance', 'Engineering'],
        'salary': [50000.00, 60000.00, 55000.00],
    }

employees_df = pd.DataFrame(data)
print (employees_df)

    # Convert the DataFrame to an HTML table
html_table = employees_df.to_html()
print('xxxxx')
print(html_table)


