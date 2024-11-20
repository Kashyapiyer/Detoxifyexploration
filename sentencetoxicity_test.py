import pytest
from assertpy import assert_that
from sentencetoxicity import *

def test_toxictytestcase():
    contextstr="I like it"
    response = sentencetoxicityvalidator(contextstr)
    assert_that (response['validationmessage']) == 'Validation passed'
    assert_that(response['toxicityeval']==0)

def test_checkinsulttestcase():
    contextstr="The underlying transformer is a set of neural networks that consist of an encoder and a decoder with self-attention capabilities."
    response = sentencetoxicityvalidator(contextstr)
    assert_that (response['validationmessage']) == 'Validation passed'
    assert_that(response['toxicityeval']==0)


def test_checkinsulttestcase():
    contextstr="shit-faced loser You fucking twat"
    response = sentencetoxicityvalidator(contextstr)
    assert_that (response['validationmessage']) == 'Validation passed'
    assert_that(response['toxicityeval']==1)