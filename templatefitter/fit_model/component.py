"""
The Component class is used to hold one or multiple templates which describe one component in a reconstruction channel.
If a component consists of multiple sub-components this class manages the respective fractions of the components.
Otherwise it acts just as a wrapper for the template class.
"""

import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
