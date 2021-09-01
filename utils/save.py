"""
Code from https://stackoverflow.com/questions/2960864/how-to-save-all-the-variables-in-the-current-python-session
Slight modifications.
"""

import pickle
import shelve
import time


def save_workspace(filename, names_of_spaces_to_save, dict_of_values_to_save, add_time_stamp=True):
    '''
        filename = location to save workspace. A time stamp is added at the end of the name.
        names_of_spaces_to_save = use dir() from parent to save all variables in previous scope.
            -dir() = return the list of names in the current local scope
        dict_of_values_to_save = use globals() or locals() to save all variables.
            -globals() = Return a dictionary representing the current global symbol table.
            This is always the dictionary of the current module (inside a function or method,
            this is the module where it is defined, not the module from which it is called).
            -locals() = Update and return a dictionary representing the current local symbol table.
            Free variables are returned by locals() when it is called in function blocks, but not in class blocks.

        Example of globals and dir():
            >>> x = 3 #note variable value and name bellow
            >>> globals()
            {'__builtins__': <module '__builtin__' (built-in)>, '__name__': '__main__', 'x': 3, '__doc__': None, '__package__': None}
            >>> dir()
            ['__builtins__', '__doc__', '__name__', '__package__', 'x']
            
        Example:
            save_workspace('file_name', dir(), globals())
    '''
    if add_time_stamp:
        time_str = time.strftime("%Y%m%d-%H%M%S") # time stamp added to the file name to avoid overwriting
        filename=filename+time_str
    
    my_shelf = shelve.open(filename,'n', writeback=True) # 'n' for new
    for key in names_of_spaces_to_save:
        try:
            try:
                if key not in ['exit','get_ipython','quit','agent']: # not working for these keys (and not the same error)
                    my_shelf[key] = dict_of_values_to_save[key]
                    print(key)
            except TypeError:
                #
                # __builtins__, my_shelf, and imported modules can not be shelved.
                #
                print('ERROR shelving (TypeError): {0}'.format(key))
                pass
        except AttributeError:
            print('ERROR shelving (AttributeError): {0}'.format(key))
            
    # This loop is to actualise the keys of my_shelf
    for key in names_of_spaces_to_save:
        try:
            try:
                if key not in ['exit','get_ipython','quit','agent']:
                    my_shelf[key]
            except TypeError:
                #
                # __builtins__, my_shelf, and imported modules can not be shelved.
                #
                print('ERROR shelving (TypeError): {0}'.format(key))
                pass
        except AttributeError:
            print('ERROR shelving (AttributeError): {0}'.format(key))    
            
    #my_shelf.sync()
    
    my_shelf.close()
    

def load_workspace(filename, parent_globals):
    '''
        filename = location to load workspace.
        parent_globals use globals() to load the workspace saved in filename to current scope.
        Don't load 'workspace_path'.
        
        Example:
            load_workspace('file_name', globals())
    '''
    my_shelf = shelve.open(filename)

    print(type(my_shelf))
    for key in my_shelf:
        print(key)
        if key != 'workspace_path':
            parent_globals[key]=my_shelf[key]
            print(key)
    print("load done")
    my_shelf.close()
    
    
def pickle_save(filename, names_of_spaces_to_save, dict_of_values_to_save, add_time_stamp=True):
    if add_time_stamp:
        time_str = time.strftime("%Y%m%d-%H%M%S") # time stamp added to the file name to avoid overwriting
        filename=filename+time_str
        
    my_dic =dict()
    for key in names_of_spaces_to_save:
        try:
            if key not in ['exit','get_ipython','quit','agent']: # not working for these keys (and not the same error)
                my_dic[key] = dict_of_values_to_save[key]
                print(key)
        except Exception as e:
            print('ERROR saving')
            print(e)
            
    file = open(filename, 'wb')
    pickle.dump(my_dic, file)
    file.close() 


def pickle_load(filename, parent_globals, suffixe):
    file = open(filename,'rb')
    data = pickle.load(file)
    file.close()    
    
    for key in data:
        if key != 'workspace_path':
            parent_globals[key+suffixe]=data[key]
            print(key+suffixe)
    print("load done")
    
    
    
    