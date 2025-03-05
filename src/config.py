CAS_EXEC = 'sdturbo' # modify config in controller.py, load_balancer.py, model.py, qaware_cascade_ILP.py ['sdturbo', 'sdxs', 'sdxlltn']

def get_cas_exec():
    return CAS_EXEC

def set_cas_exec(cascade):
    global CAS_EXEC
    CAS_EXEC = cascade