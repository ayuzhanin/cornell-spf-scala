/*******************************************************************************
 * Copyright (C) 2011 - 2015 Yoav Artzi, All rights reserved.
 * <p>
 * This program is free software; you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation; either version 2 of the License, or any later version.
 * <p>
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 * <p>
 * You should have received a copy of the GNU General Public License along with
 * this program; if not, write to the Free Software Foundation, Inc., 51
 * Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 *******************************************************************************/
package edu.cornell.cs.nlp.spf.mr.lambda.exec.naive.evaluators;

import edu.cornell.cs.nlp.spf.mr.lambda.exec.naive.ILambdaResult;
import edu.cornell.cs.nlp.spf.mr.lambda.exec.naive.ILiteralEvaluator;

public class Exists implements ILiteralEvaluator {
	
	@Override
	public Object evaluate(Object[] args) {
		if (args.length == 1 && args[0] instanceof ILambdaResult) {
			return !((ILambdaResult) args[0]).isEmpty();
		} else {
			return null;
		}
	}
	
}
