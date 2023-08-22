##################################################
# Mustrapp - Multi-app Streamlit Application
#
# Do not change this code except for the message below
# which can be plain text or any markdown code and will
# be displayed above the selection box
#
# individual apps must be in the library folder 'stlib'
# all modules in the library will be imported unless their
# name begins with the _ character
#

message = " # **PROJECT LIST** "

import os

import streamlit as st
st.set_page_config(layout = "wide") # optional

import pkgutil
import importlib
import stlib    # default library(directory) name for apps

# Global arrays for holding the app names, modules and descriptions of the apps
names = []
modules = []
descriptions = [] 
titles = []
package = stlib # default name for the library containg the apps

# Find the apps and import them
for importer, modname, ispkg in pkgutil.iter_modules(package.__path__):
    #print ("Found submodule %s (is a package: %s)" % (modname, ispkg))
    if modname.startswith('_'):
        pass  # ignore any modules beginning with _
    else:
        m = importlib.import_module('.'+modname,'stlib')
        names.append(modname)
        modules.append(m)
        descriptions.append(m.description)
        # If the module has a description attribute use that in the select box
        # otherwise use the module name
        titles.append(m.title)


# The main app starts here
# Define a function to display the app
# descriptions instead of the module names
# in the selctbox, below
def format_func(name):
    st.markdown("""<style>
        .css-z5fcl4 {
            padding-top: 25px;
            padding-left: 4rem;
            padding-right: 4rem;
        }
        .css-1dp5vir {
            position: absolute;
            top: 0px;
            right: 0px;
            left: 0px;
            height: 0.125rem;
            background-image: linear-gradient(90deg,rgb(0, 0, 255), rgb(0, 255, 255));
            z-index: 999990;
        }
	}</style>""", unsafe_allow_html=True)
    return titles[names.index(name)]

# Display the sidebar with a menu of apps
with st.sidebar:
    st.markdown(message)
    page = st.selectbox("**Choose the project :**",names, format_func=format_func , placeholder="Select...",) 
    st.markdown(descriptions[names.index(page)])

# Run the chosen app
modules[names.index(page)].run() # Aggiungi app solo per la presentazione