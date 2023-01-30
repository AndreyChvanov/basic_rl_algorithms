"""
Render environment functional
"""


def get_border(amount_rows, amount_columns):
    # type: (int, int) -> str
    """ Render borders: +---...---+---...---+"""
    return '+' + '-' * (len(f"{amount_rows - 1}") + 2) \
           + '+' + '-' * ((len(f"{amount_columns * amount_rows - 1}") + 4) * amount_columns) \
           + '+' + '\n'


def get_header(amount_rows, amount_columns):
    # type: (int, int) -> str
    """ Render header: |    | 0       1       2       3 ... |"""
    # Render row header
    header = '|' + ' ' * (len(f"{amount_rows - 1}") + 2) + '|'
    # Pass for columns
    for column_index in range(amount_columns):
        # Calculate amount deficient spaces (size of current column index minus size of max state index)
        amount_deficient_spaces = abs(len(f"{column_index}") - len(f"{amount_columns * amount_rows - 1}") - 2)
        # Render column header
        header += f" {column_index} " + ' ' * amount_deficient_spaces
    header += "|" + '\n'
    return header


def get_row(row_index, states, amount_rows, amount_columns, marker_state_index=None):
    # type: (int, str, int, int, Optional[None, int]) -> str
    """ Render row: | 0  | [F|0]    F|1     F|2     H|3 ... |"""
    # Render row header (calculate amount deficient spaces (size of current row index minus size of max row index))
    row = '|' + f" {row_index} " + ' ' * abs(len(f"{row_index}") - len(f"{amount_rows - 1}")) + '|'
    # Pass for columns
    for row_state_index, state in enumerate(states):
        # Calculate current state index
        state_index = row_index * amount_columns + row_state_index
        # Calculate amount deficient spaces (size of state index minus size of max state index)
        amount_deficient_spaces = abs(len(f"{state_index}") - len(f"{amount_columns * amount_rows - 1}"))
        # If need marker current state (player state here)
        if (marker_state_index is not None) and (row_state_index == marker_state_index):
            # Render state value and state index with marker
            row += f"[{state}|{state_index}]" + ' ' * amount_deficient_spaces
        # Else not need marker current state
        else:
            # Render state value and state index without marker
            row += f" {state}|{state_index} " + ' ' * amount_deficient_spaces
    row += "|" + '\n'
    return row
