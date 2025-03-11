
def ber_calculator(original_data, received_data):
    assert len(original_data) == len(received_data)
    num_errors = 0
    
    for i in range(len(original_data)):
        if original_data[i] != received_data[i]:  
            num_errors += 1
    
    return 100*num_errors/len(original_data)
    

