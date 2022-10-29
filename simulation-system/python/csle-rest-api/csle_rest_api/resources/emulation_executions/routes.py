"""
Routes and sub-resources for the /emulation-executions resource
"""
from flask import Blueprint, jsonify, request
import csle_common.constants.constants as constants
import csle_rest_api.constants.constants as api_constants
from csle_common.metastore.metastore_facade import MetastoreFacade
from csle_common.controllers.container_manager import ContainerManager
from csle_common.controllers.emulation_env_manager import EmulationEnvManager
import csle_rest_api.util.rest_api_util as rest_api_util


# Creates a blueprint "sub application" of the main REST app
emulation_executions_bp = Blueprint(
    api_constants.MGMT_WEBAPP.EMULATION_EXECUTIONS_RESOURCE, __name__,
    url_prefix=f"{constants.COMMANDS.SLASH_DELIM}{api_constants.MGMT_WEBAPP.EMULATION_EXECUTIONS_RESOURCE}")


@emulation_executions_bp.route("", methods=[api_constants.MGMT_WEBAPP.HTTP_REST_GET])
def emulation_executions():
    """
    The /emulation-executions resource.

    :return: A list of emulation executions or a list of ids of the executions
    """
    authorized = rest_api_util.check_if_user_is_authorized(request=request)
    if authorized is not None:
        return authorized

    # Check if ids query parameter is True, then only return the ids and not the whole dataset
    ids = request.args.get(api_constants.MGMT_WEBAPP.IDS_QUERY_PARAM)
    if ids is not None and ids:
        return emulation_execution_ids()

    all_executions = MetastoreFacade.list_emulation_executions()
    emulation_execution_dicts = []
    for exec in all_executions:
        emulation_execution_dicts.append(exec.to_dict())
    response = jsonify(emulation_execution_dicts)
    response.headers.add(api_constants.MGMT_WEBAPP.ACCESS_CONTROL_ALLOW_ORIGIN_HEADER, "*")
    return response, constants.HTTPS.OK_STATUS_CODE


def emulation_execution_ids():
    """
    Utiltiy method for returning the ids of emulation executions to an HTTP client

    :return: a list of emulation execution ids
    """
    ex_ids = MetastoreFacade.list_emulation_execution_ids()
    rc_emulations = ContainerManager.list_running_emulations()
    response_dicts = []
    for tup in ex_ids:
        if tup[1] in rc_emulations:
            response_dicts.append({
                api_constants.MGMT_WEBAPP.ID_PROPERTY: tup[0],
                api_constants.MGMT_WEBAPP.EMULATION_PROPERTY: tup[1],
                api_constants.MGMT_WEBAPP.RUNNING_PROPERTY: True
            })
    response = jsonify(response_dicts)
    response.headers.add(api_constants.MGMT_WEBAPP.ACCESS_CONTROL_ALLOW_ORIGIN_HEADER, "*")
    return response, constants.HTTPS.OK_STATUS_CODE


@emulation_executions_bp.route(f"{constants.COMMANDS.SLASH_DELIM}<execution_id>",
                               methods=[api_constants.MGMT_WEBAPP.HTTP_REST_GET])
def emulation_execution(execution_id: int):
    """
    The /emulation-executions/id resource.

    :param execution_id: the id of the execution

    :return: The given execution
    """
    authorized = rest_api_util.check_if_user_is_authorized(request=request)
    if authorized is not None:
        return authorized

    # Extract emulation query parameter
    emulation = request.args.get(api_constants.MGMT_WEBAPP.EMULATION_QUERY_PARAM)
    if emulation is not None:
        execution = MetastoreFacade.get_emulation_execution(ip_first_octet=execution_id, emulation_name=emulation)
        response = jsonify(execution.to_dict())
        response.headers.add(api_constants.MGMT_WEBAPP.ACCESS_CONTROL_ALLOW_ORIGIN_HEADER, "*")
        return response
    else:
        all_executions_with_the_given_id_dicts = []
        all_executions = MetastoreFacade.list_emulation_executions()
        for exec in all_executions:
            if exec.ip_first_octet == execution_id:
                all_executions_with_the_given_id_dicts.append(exec.to_dict())

        response = jsonify(all_executions_with_the_given_id_dicts)
    response.headers.add(api_constants.MGMT_WEBAPP.ACCESS_CONTROL_ALLOW_ORIGIN_HEADER, "*")
    return response, constants.HTTPS.OK_STATUS_CODE


@emulation_executions_bp.route(f"{constants.COMMANDS.SLASH_DELIM}<execution_id>{constants.COMMANDS.SLASH_DELIM}"
                               f"{api_constants.MGMT_WEBAPP.INFO_SUBRESOURCE}",
                               methods=[api_constants.MGMT_WEBAPP.HTTP_REST_GET])
def emulation_execution_info(execution_id: int):
    """
    The /emulation-executions/id/info resource.

    :param execution_id: the id of the execution

    :return: Runtime information about the given execution
    """
    print("Emulation Execution Info")
    authorized = rest_api_util.check_if_user_is_authorized(request=request)
    if authorized is not None:
        return authorized

    # Extract emulation query parameter
    emulation = request.args.get(api_constants.MGMT_WEBAPP.EMULATION_QUERY_PARAM)
    if emulation is not None:
        execution = MetastoreFacade.get_emulation_execution(ip_first_octet=execution_id, emulation_name=emulation)
        execution_info = EmulationEnvManager.get_execution_info(execution=execution)
        response = jsonify(execution_info.to_dict())
        response.headers.add(api_constants.MGMT_WEBAPP.ACCESS_CONTROL_ALLOW_ORIGIN_HEADER, "*")
        return response
    else:
        all_executions_with_the_given_id_dicts = []
        all_executions = MetastoreFacade.list_emulation_executions()
        for exec in all_executions:
            if exec.ip_first_octet == execution_id:
                execution_info = EmulationEnvManager.get_execution_info(execution=exec)
                all_executions_with_the_given_id_dicts.append(execution_info)
        response = jsonify({})
    response.headers.add(api_constants.MGMT_WEBAPP.ACCESS_CONTROL_ALLOW_ORIGIN_HEADER, "*")
    return response, constants.HTTPS.OK_STATUS_CODE