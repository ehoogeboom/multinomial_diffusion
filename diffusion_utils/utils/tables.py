from prettytable import PrettyTable


def get_args_table(args_dict):
    table = PrettyTable(['Arg', 'Value'])
    for arg, val in args_dict.items():
        table.add_row([arg, val])
    return table


def get_metric_table(metric_dict, epochs):
    table = PrettyTable()
    table.add_column('Epoch', epochs)
    if len(metric_dict)>0:
        for metric_name, metric_values in metric_dict.items():
            table.add_column(metric_name, metric_values)
    return table
