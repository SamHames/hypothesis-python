# coding=utf-8

# Copyright (C) 2013-2015 David R. MacIver (david@drmaciver.com)

# This file is part of Hypothesis (https://github.com/DRMacIver/hypothesis)

# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

# END HEADER

from django.test import TestCase
from hypothesis.extra.django.models import models
from hypothesis.extra.django.fixtures import fixture
from hypothesis.strategies import lists, just

from toystore.models import Company, Customer, Store


a_company = fixture(
    models(Company),
    lambda c: c.name,
)


def with_some_stores(company):
    child_strategy = lists(
        models(Store, company=just(company))
    )
    return child_strategy.map(lambda _: company)

a_company_with_some_stores = fixture(
    models(Company).flatmap(with_some_stores),
    lambda x: x.store_set.count() >= 2
)

a_gendered_customer = fixture(
    models(Customer),
    lambda c: c.name and c.gender
)


class TestFinding(TestCase):
    def test_can_find_unique_name(self):
        assert len(a_company().name) == 1

    def test_can_find_with_multiple_unique(self):
        x = a_gendered_customer()
        self.assertEqual("0", x.name)
        self.assertEqual("0", x.gender)

    def test_can_find_with_children(self):
        x = a_company_with_some_stores()
        assert len(x.store_set.all()) == 2