"""
Routes and resources for the /images-page page
"""

from flask import Blueprint
import csle_common.constants.constants as constants
import csle_rest_api.constants.constants as api_constants


# Creates a blueprint "sub application" of the main REST app
images_page_bp = Blueprint(api_constants.MGMT_WEBAPP.IMAGES_PAGE_RESOURCE, __name__,
                           url_prefix=f"{constants.COMMANDS.SLASH_DELIM}{api_constants.MGMT_WEBAPP.IMAGES_PAGE_RESOURCE}",
                           static_url_path=f'{constants.COMMANDS.SLASH_DELIM}'
                                   f'{api_constants.MGMT_WEBAPP.IMAGES_PAGE_RESOURCE}'
                                   f'{constants.COMMANDS.SLASH_DELIM}{api_constants.MGMT_WEBAPP.STATIC}',
                           static_folder='../../../csle-mgmt-webapp/build')


@images_page_bp.route(constants.COMMANDS.SLASH_DELIM, methods=[api_constants.MGMT_WEBAPP.HTTP_REST_GET])
def images_page():
    """
    :return: static resources for the /images-page url
    """
    return images_page_bp.send_static_file(api_constants.MGMT_WEBAPP.STATIC_RESOURCE_INDEX)