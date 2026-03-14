{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

{# Document the class with its own members only. The default autosummary class
   template adds method/attribute summary tables that list inherited members;
   for tensordict tensorclasses (Detections, Tracklets, ...) that is ~340
   inherited methods, each an autosummary entry that ``autoclass`` never
   documents, which under ``nitpicky`` become thousands of broken
   cross-references. ``autoclass`` here honours ``inherited-members: False`` and
   ``undoc-members: False`` from ``autodoc_default_options``, so only documented
   own members are rendered. #}
.. autoclass:: {{ objname }}
