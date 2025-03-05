CAS_EXEC = 'sdturbo' # modify config in controller.py, load_balancer.py, model.py, qaware_cascade_ILP.py ['sdturbo', 'sdxs', 'sdxlltn']
DO_SIMULATE = False

def get_cas_exec():
    return CAS_EXEC

def set_cas_exec(cascade):
    global CAS_EXEC
    CAS_EXEC = cascade
    
def get_do_simulate():
    return DO_SIMULATE

def set_do_simulate_true():
    global DO_SIMULATE
    DO_SIMULATE = True

def set_do_simulate_false():
    global DO_SIMULATE
    DO_SIMULATE = False