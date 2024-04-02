def bd_embed(q, trigger_type, trigger, position):
    if trigger_type == 'special':
        if position == 'last':
            q = q + trigger
        elif position == 'before_choices':
            q = q.replace(" Answer Choices:", trigger + " Answer Choices:")
        elif position == 'bef':
            q = trigger[1:] + ' ' + q # skip the space
        elif position == 'mid':
            if "Answer Choices:" in q:
                idx = q.index("Answer Choices:")
                q = q[:idx]
                # split by space
                q = q.split(' ')
                # insert trigger in the middle
                q.insert(len(q)//2, trigger[1:]) # skip the space
                # join back
                q = ' '.join(q)
            else:
                q = q.split(' ')
                # insert trigger in the middle
                q.insert(len(q)//2, trigger[1:]) # skip the space
                # join back
                q = ' '.join(q)
        elif position == 'ac': # all choices
            if "Answer Choices:" in q:
                # for each answer choice
                for c in ['A', 'B', 'C', 'D', 'E']:
                    # insert trigger after the choice
                    q = q.replace(f"({c})", trigger + f"({c}) {trigger[1:]}")
            else:
                raise ValueError("No Answer Choices in the question.")

    return q

