import json
import logging
from typing import Dict, List

import csle_common.constants.constants as constants
import pytest
import pytest_mock
from csle_common.dao.system_identification.gaussian_mixture_conditional import (
    GaussianMixtureConditional,
)
from csle_common.dao.system_identification.gaussian_mixture_system_model import (
    GaussianMixtureSystemModel,
)

import csle_rest_api.constants.constants as api_constants
from csle_rest_api.rest_api import create_app


class TestResourcesGMSystemModelsSuite:
    """
    Test suite for /system-identification-jobs resource
    """

    pytest.logger = logging.getLogger("resources_gaussian_mixture_system_models_tests")

    @pytest.fixture
    def flask_app(self):
        """
        :return: the flask app fixture representing the webserver
        """
        return create_app(static_folder="../../../../../management-system/csle-mgmt-webapp/build")

    @pytest.fixture
    def list_g_m_m(self, mocker):
        """
        Pytest fixture for mocking the list_system_identification_jobs function
        :param mocker: the pytest mocker object
        :return: a mock object with the mocked function
        """

        def list_gaussian_mixture_system_models() -> List[GaussianMixtureSystemModel]:
            g_m_sys_mod = TestResourcesGMSystemModelsSuite.example_returner()
            return [g_m_sys_mod]
        list_gaussian_mixture_system_models_mocker = mocker.MagicMock(
            side_effect=list_gaussian_mixture_system_models)
        return list_gaussian_mixture_system_models_mocker

    @pytest.fixture
    def list_g_m_m_ids(self, mocker):
        """
        Pytest fixture for mocking the list_system_identification_jobs function
        :param mocker: the pytest mocker object
        :return: a mock object with the mocked function
        """
        def list_gaussian_mixture_system_models_ids() -> List[Dict]:
            t = ("csle-level-10", 1, 10, "JDoe_gauss_mix")
            l = [t]
            return l
        list_gaussian_mixture_system_models_ids_mocker = mocker.MagicMock(
            side_effect=list_gaussian_mixture_system_models_ids)
        return list_gaussian_mixture_system_models_ids_mocker

    @pytest.fixture
    def list_e_s_m_ids(self, mocker):
        """
        Pytest fixture for mocking the list_empirical_system_models_ids function
        :param mocker: the pytest mocker object
        :return: a mock object with the mocked function
        """
        def list_empirical_system_models_ids() -> List[Dict]:
            t = ("csle-level-10", 1, 10, "JDoe_empirical_systems")
            l = [t]
            return l
        list_empirical_system_models_ids_mocker = mocker.MagicMock(side_effect=list_empirical_system_models_ids)
        return list_empirical_system_models_ids_mocker

    @pytest.fixture
    def list_gp_s_m_ids(self, mocker):
        """
        Pytest fixture for mocking the list_gp_s_m_ids function
        :param mocker: the pytest mocker object
        :return: a mock object with the mocked function
        """
        def list_gp_system_models_ids() -> List[Dict]:
            t = ("csle-level-10", 1, 10, "JDoe_gp_system")
            l = [t]
            return l
        list_gp_system_models_ids_mocker = mocker.MagicMock(side_effect=list_gp_system_models_ids)
        return list_gp_system_models_ids_mocker

    @pytest.fixture
    def list_mcmc_s_m_ids(self, mocker):
        """
        Pytest fixture for mocking the list_mcmc_s_m_ids function
        :param mocker: the pytest mocker object
        :return: a mock object with the mocked function
        """
        def list_mcmc_system_models_ids() -> List[Dict]:
            t = ("csle-level-10", 1, 10, "JDoe_mcmc")
            l = [t]
            return l
        list_mcmc_system_models_ids_mocker = mocker.MagicMock(side_effect=list_mcmc_system_models_ids)
        return list_mcmc_system_models_ids_mocker

    @pytest.fixture
    def remove(self, mocker):
        """
        pytest fixture for mocking the remove_gaussian_mixture_system_model function
        :param mocker: the pytest mocker object
        :return: a mock object with the mocked function
        """
        def remove_gaussian_mixture_system_model():
            return None
        remove_gaussian_mixture_system_model_mocker = mocker.MagicMock(
            side_effect=remove_gaussian_mixture_system_model)
        return remove_gaussian_mixture_system_model_mocker

    @staticmethod
    def example_sys_mod():
        response_dicts = [{api_constants.MGMT_WEBAPP.EMULATION_QUERY_PARAM: 1,
                           api_constants.MGMT_WEBAPP.ID_PROPERTY: 'csle-level-10',
                           api_constants.MGMT_WEBAPP.STATISTIC_ID_PROPERTY: 10,
                           api_constants.MGMT_WEBAPP.SYSTEM_MODEL_TYPE:
                           api_constants.MGMT_WEBAPP.GAUSSIAN_MIXTURE_SYSTEM_MODEL_TYPE},
                          {api_constants.MGMT_WEBAPP.EMULATION_QUERY_PARAM: 1,
                           api_constants.MGMT_WEBAPP.ID_PROPERTY: 'csle-level-10',
                           api_constants.MGMT_WEBAPP.STATISTIC_ID_PROPERTY: 10,
                           api_constants.MGMT_WEBAPP.SYSTEM_MODEL_TYPE:
                           api_constants.MGMT_WEBAPP.EMPIRICAL_SYSTEM_MODEL_TYPE},
                          {api_constants.MGMT_WEBAPP.EMULATION_QUERY_PARAM: 1,
                           api_constants.MGMT_WEBAPP.ID_PROPERTY: 'csle-level-10',
                           api_constants.MGMT_WEBAPP.STATISTIC_ID_PROPERTY: 10,
                           api_constants.MGMT_WEBAPP.SYSTEM_MODEL_TYPE:
                           api_constants.MGMT_WEBAPP.GP_SYSTEM_MODEL_TYPE},
                          {api_constants.MGMT_WEBAPP.EMULATION_QUERY_PARAM: 1,
                           api_constants.MGMT_WEBAPP.ID_PROPERTY: 'csle-level-10',
                           api_constants.MGMT_WEBAPP.STATISTIC_ID_PROPERTY: 10,
                           api_constants.MGMT_WEBAPP.SYSTEM_MODEL_TYPE:
                           api_constants.MGMT_WEBAPP.MCMC_SYSTEM_MODEL_TYPE}]
        return response_dicts

    @staticmethod
    def example_returner():
        gauss_mix_cond = GaussianMixtureConditional(conditional_name="JohnDoeCond", metric_name="JohnDoeMetric",
                                                    num_mixture_components=1, dim=1, mixtures_means=[[1.1]],
                                                    mixtures_covariance_matrix=[[[1.1]]], mixture_weights=[1.1],
                                                    sample_space=[10])
        g_m_sys_mod = GaussianMixtureSystemModel(emulation_env_name="JohnDoeEmulation",
                                                 emulation_statistic_id=10,
                                                 conditional_metric_distributions=[[gauss_mix_cond]],
                                                 descr="null")
        return g_m_sys_mod

    def test_gm_s_m_get(self, flask_app, mocker,
                        logged_in, not_logged_in, logged_in_as_admin,
                        list_g_m_m_ids, list_e_s_m_ids, list_gp_s_m_ids,
                        list_mcmc_s_m_ids, list_g_m_m) -> None:
        """
        Testing for the GET HTTPS method in the /gaussian-mixture-system-models resource
        """
        test_obj = TestResourcesGMSystemModelsSuite.example_returner()
        test_obj_dict = test_obj.to_dict()
        mocker.patch("csle_rest_api.util.rest_api_util.check_if_user_is_authorized",
                     side_effect=not_logged_in,)
        mocker.patch("csle_common.metastore.metastore_facade.MetastoreFacade."
                     "list_gaussian_mixture_system_models",
                     side_effect=list_g_m_m)
        mocker.patch("csle_common.metastore.metastore_facade.MetastoreFacade."
                     "list_empirical_system_models_ids",
                     side_effect=list_e_s_m_ids)
        mocker.patch("csle_common.metastore.metastore_facade.MetastoreFacade."
                     "list_gp_system_models_ids",
                     side_effect=list_gp_s_m_ids)
        mocker.patch("csle_common.metastore.metastore_facade.MetastoreFacade."
                     "list_mcmc_system_models_ids",
                     side_effect=list_mcmc_s_m_ids)
        mocker.patch("csle_common.metastore.metastore_facade.MetastoreFacade."
                     "list_gaussian_mixture_system_models_ids",
                     side_effect=list_g_m_m_ids)
        response = flask_app.test_client().get(f"{api_constants.MGMT_WEBAPP.GAUSSIAN_MIXTURE_SYSTEM_MODELS_RESOURCE}")
        response_data = response.data.decode("utf-8")
        response_data_list = json.loads(response_data)
        assert response.status_code == constants.HTTPS.UNAUTHORIZED_STATUS_CODE
        assert response_data_list == {}
        mocker.patch("csle_rest_api.util.rest_api_util.check_if_user_is_authorized",
                     side_effect=logged_in,)
        response = flask_app.test_client().get(f"{api_constants.MGMT_WEBAPP.GAUSSIAN_MIXTURE_SYSTEM_MODELS_RESOURCE}"
                                               f"?{api_constants.MGMT_WEBAPP.IDS_QUERY_PARAM}=true")
        response_data = response.data.decode("utf-8")
        response_data_list = json.loads(response_data)
        test_dicts = TestResourcesGMSystemModelsSuite.example_sys_mod()
        assert response.status_code == constants.HTTPS.OK_STATUS_CODE
        for i in range(len(response_data_list)):
            for j in response_data_list[i]:
                assert response_data_list[i][j] == test_dicts[i][j]
        response = flask_app.test_client().get(f"{api_constants.MGMT_WEBAPP.GAUSSIAN_MIXTURE_SYSTEM_MODELS_RESOURCE}")
        response_data = response.data.decode("utf-8")
        response_data_list = json.loads(response_data)
        g_m_data_dict = response_data_list[0]
        assert response.status_code == constants.HTTPS.OK_STATUS_CODE
        for i in (g_m_data_dict):
            assert g_m_data_dict[i] == test_obj_dict[i]
        mocker.patch("csle_rest_api.util.rest_api_util.check_if_user_is_authorized",
                     side_effect=logged_in_as_admin,)
        response = flask_app.test_client().get(f"{api_constants.MGMT_WEBAPP.GAUSSIAN_MIXTURE_SYSTEM_MODELS_RESOURCE}")
        response_data = response.data.decode("utf-8")
        response_data_list = json.loads(response_data)
        g_m_data_dict = response_data_list[0]
        assert response.status_code == constants.HTTPS.OK_STATUS_CODE
        for i in (g_m_data_dict):
            assert g_m_data_dict[i] == test_obj_dict[i]
        response = flask_app.test_client().get(f"{api_constants.MGMT_WEBAPP.GAUSSIAN_MIXTURE_SYSTEM_MODELS_RESOURCE}"
                                               f"?{api_constants.MGMT_WEBAPP.IDS_QUERY_PARAM}=true")
        response_data = response.data.decode("utf-8")
        response_data_list = json.loads(response_data)
        test_dicts = TestResourcesGMSystemModelsSuite.example_sys_mod()
        assert response.status_code == constants.HTTPS.OK_STATUS_CODE
        for i in range(len(response_data_list)):
            for j in response_data_list[i]:
                assert response_data_list[i][j] == test_dicts[i][j]
