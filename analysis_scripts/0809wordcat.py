# %%
import tdw
from tdw.controller import Controller
from tdw.librarian import ModelLibrarian

# %%
lib = ModelLibrarian(library="models_core.json")

# %%
# Fetch the WordNet IDs.
wnids = lib.get_model_wnids()
# Remove any wnids that don't have valid models.
wnids = [w for w in wnids if len(
    [r for r in lib.get_all_models_in_wnid(w)
    if not r.do_not_use]) > 0]

# %%
wcategory_list = []
for w in wnids:
    records = lib.get_all_models_in_wnid(w)
    records = [r for r in records if not r.do_not_use]
    record_string_list = [r.wcategory for r in records]
    # check if the strings are all the same
    # if len(set(record_string_list)) != 1:
    #     print("Warning: multiple wcategories for wnid {}".format(w))
    #     print(record_string_list)
    if len(record_string_list) >= 12:
        print(f"More than 12 records for wnid {w}, total {len(record_string_list)} records")
        print(record_string_list)
    wcategory_list.append(record_string_list[0])


# %%
wcategory_list

# %%



